import os, sys, time
import math
import io, queue, threading
from pprint import pprint

import numpy as np
import torch, torchvision
from pycocotools.cocoeval import COCOeval

import tensorrt as trt
import onnx
from onnx import shape_inference, helper, TensorProto
import onnx_graphsurgeon as gs

import pycuda.driver as cuda
import pycuda.autoinit

from data import get_val_dataloader, get_coco_ground_truth, init_dboxes
import gpuplot


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, args):
        super().__init__()
        self.batch_dim = args.batch_dim
        self.dataloader = iter(get_val_dataloader(args))
        self.current_batch = None # for ref-counting
        self.cache_path = 'calibration.cache'

    def get_batch_size(self):
        return self.batch_dim

    def get_batch(self, tensor_names):
        # assume same order as in dataset
        try:
            tensor_nchw, _, heights_widths, _, r_e = next(self.dataloader)
            self.current_batch = tensor_nchw.cuda(), heights_widths[0].cuda(), heights_widths[1].cuda()
            return [t.data_ptr() for t in self.current_batch]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_path, 'wb') as f:
            f.write(cache)



class SSD300(torch.nn.Module):
    def __init__(self, topk, detection_threshold, iou_threshold, model_precision, batch_dim, trt_path=None, onnx_export=False):
        super().__init__()
        self.topk = torch.nn.Parameter(torch.tensor(topk, dtype=torch.int32), requires_grad=False)
        self.detection_threshold = torch.nn.Parameter(torch.tensor(detection_threshold), requires_grad=False)
        self.model_dtype = torch.float16 if model_precision == 'fp16' else torch.float32
        self.batch_dim = batch_dim
        self.class_dim = 81
        self.foreground_class_dim = self.class_dim - 1
        self.scale_xy = 0.1
        self.scale_wh = 0.2
        self.scale_xyxywhwh = torch.nn.Parameter(torch.tensor([
            self.scale_xy,
            self.scale_xy,
            self.scale_wh,
            self.scale_wh
        ]), requires_grad=False)
        self.scale_wh_delta = torch.nn.Parameter(torch.tensor([-0.5, -0.5, 0.5, 0.5]), requires_grad=False)
        self.iou_threshold = iou_threshold
        self.dboxes_xywh = torch.nn.Parameter(init_dboxes(self.model_dtype).unsqueeze(dim=0), requires_grad=False)
        self.box_dim = torch.nn.Parameter(torch.tensor(self.dboxes_xywh.size(1)), requires_grad=False)
        self.buffer_nchw = torch.nn.Parameter(torch.zeros((batch_dim, 3, 300, 300), dtype=self.model_dtype), requires_grad=False)
        self.class_indexes = torch.nn.Parameter(torch.arange(1, self.class_dim).repeat(self.batch_dim * self.topk), requires_grad=False)
        self.image_indexes = torch.nn.Parameter(
            (torch.ones(self.topk * self.foreground_class_dim, dtype=torch.int32) * torch.arange(self.batch_dim).unsqueeze(-1)).view(-1),
            requires_grad=False
        )
        self.onnx_export = onnx_export
        self.trt_engine = None
        if trt_path:
            print('loading TRT engine from', trt_path, '...')
            self.trt_logger = trt.Logger()
            trt.init_libnvinfer_plugins(self.trt_logger, '')
            with open(trt_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
                self.trt_engine = runtime.deserialize_cuda_engine(f.read())
                self.trt_stream = cuda.Stream()
                self.trt_context = self.trt_engine.create_execution_context()
        else:
            self.detector = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=model_precision).eval()


    def forward(self, tensor_nchw, image_heights, image_widths):
        if self.onnx_export:
            return self.forward_pytorch(tensor_nchw, image_heights, image_widths)
        else:
            return self.forward_coco(tensor_nchw, image_heights, image_widths)


    def forward_pytorch(self, tensor_nchw, image_heights, image_widths):
        locs, scores = self.detector(tensor_nchw)
        locs = locs.permute(0, 2, 1)
        locs = self.rescale_locs(locs)

        scores = scores.permute(0, 2, 1)
        probs = torch.nn.functional.softmax(scores, dim=-1)

        locs, probs = self.reshape_for_topk(locs, probs)
        bboxes = self.locs_to_xyxy(locs, image_heights, image_widths)
        return bboxes, probs


    def forward_trt(self, tensor_nchw, image_heights, image_widths):
        trt_outputs, bindings = [], []
        np_to_torch_type = {
            np.float32: torch.float32,
            np.float16: torch.float16,
            np.int32: torch.int32,
            np.int64: torch.int64,
        }

        for binding_name in self.trt_engine:
            shape = self.trt_engine.get_tensor_shape(binding_name)
            shape = [tensor_nchw.size(0) if dim == -1 else dim for dim in shape]
            dtype = trt.nptype(self.trt_engine.get_tensor_dtype(binding_name))
            torch_type = np_to_torch_type[dtype]

            if self.trt_engine.get_tensor_mode(binding_name) and (binding_name != "SelectedIndices") and (binding_name != "NumOutputBoxes") and (binding_name != "bboxes") and (binding_name != "probs"):
                torch_input = vars()[binding_name].to(torch_type)
                bindings.append(int(torch_input.data_ptr()))
            else:
                torch_output = torch.zeros(tuple(shape), dtype=torch_type, device='cuda')
                trt_outputs.append(torch_output)
                bindings.append(int(torch_output.data_ptr()))

        for i in range(self.trt_engine.num_io_tensors):
            self.trt_context.set_tensor_address(self.trt_engine.get_tensor_name(i), bindings[i])

        self.trt_context.execute_async_v3(stream_handle=self.trt_stream.handle)
        self.trt_stream.synchronize()

        return trt_outputs


    def trt_postprocess(self, batch_dim, num_detections, bboxes, probs, class_indexes):
        # select valid detections and flatten batch/box/class dimensions
        num_detections = num_detections.expand(-1, self.topk)
        detection_mask = num_detections > torch.arange(self.topk, dtype=torch.int32, device='cuda').unsqueeze(0).expand(-1, self.topk)

        probs = probs.masked_select(detection_mask)
        class_indexes = self.class_indexes[class_indexes.to(torch.int64)].masked_select(detection_mask)

        image_indexes = torch.arange(batch_dim, dtype=torch.int64, device='cuda').unsqueeze(-1).expand(-1, self.topk)
        image_indexes = image_indexes.masked_select(detection_mask)

        bboxes = bboxes.masked_select(detection_mask.unsqueeze(-1).expand_as(bboxes))
        bboxes = bboxes.unsqueeze(-1).reshape(-1, 4)

        return bboxes, probs, class_indexes, image_indexes

    def trt_postprocess_inms(self, selected_indices, num_output_boxes, bboxes, probs):
        # selected_indices: [Num_output_boxes, 3] (batchIndex, classIndex, boxIndex)
        # bboxes: [batch, anchors, classes, 4]
        # probs: [batch, anchors, classes]

        # Separate batch_index, class_index, box_index in the selected_indices
        batch_inds, cls_inds, box_inds = selected_indices.unbind(1)

        # Select boxes and scores based on selected_indices
        selected_boxes = bboxes[batch_inds, box_inds, cls_inds, :]  # shape: [Num_output_boxes, 4]
        selected_scores = probs[batch_inds, box_inds, cls_inds]     # shape: [Num_output_boxes]

        # Create class_indexes and image_indexes. In here, tensorrt output class started from 0, but coco2017 started from 1 so add 1
        class_indexes = cls_inds + 1
        image_indexes = batch_inds

        # prepare returns after NMS
        flat_locs = selected_boxes  # coordinate of detected boxes
        flat_probs = selected_scores  # scores of detected boxes

        return flat_locs, flat_probs, class_indexes, image_indexes

    def forward_coco(self, tensor_nchw, image_heights, image_widths):
        if self.trt_engine:
            trt_outputs = self.forward_trt(tensor_nchw, image_heights, image_widths)
            bboxes, probs, class_indexes, image_indexes = self.trt_postprocess_inms(
                trt_outputs[0], trt_outputs[1], trt_outputs[2], trt_outputs[3]
            )
        else:
            bboxes, probs = self.forward_pytorch(tensor_nchw, image_heights, image_widths)
            bboxes, probs, class_indexes, image_indexes = self.topk_and_nms(bboxes, probs)
        return self.xyxy_to_xywh(bboxes), probs, class_indexes, image_indexes


    def rescale_locs(self, locs):
        locs *= self.scale_xyxywhwh

        xy = locs[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        wh = locs[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        wh_delta = torch.cat([wh, wh], dim=-1) * self.scale_wh_delta
        cxycxy = torch.cat([xy, xy], dim=-1)
        return cxycxy + wh_delta


    def reshape_for_topk(self, locs, probs):
        locs = locs.unsqueeze(-2)
        locs = locs.expand(locs.size(0), self.box_dim, self.foreground_class_dim, locs.size(3))
        probs = probs[:, :, 1:]
        return locs, probs


    def topk_and_nms(self, locs, probs):
        probs, top_prob_indexes = probs.topk(self.topk, dim=1)
        flat_probs = probs.reshape(-1).contiguous()

        locs = locs.gather(1, top_prob_indexes.unsqueeze(-1).expand(*top_prob_indexes.size(), 4))
        flat_locs = locs.reshape(-1, 4).contiguous()

        # only do NMS on detections over threshold
        threshold_mask = flat_probs > self.detection_threshold

        flat_locs = flat_locs[threshold_mask]
        flat_probs = flat_probs[threshold_mask]
        class_indexes = self.class_indexes[threshold_mask]
        image_indexes = self.image_indexes[threshold_mask]

        nms_mask = torchvision.ops.boxes.batched_nms(
            flat_locs,
            flat_probs,
            class_indexes * (image_indexes + 1), # do not multiply class_indexes by 0
            iou_threshold=self.iou_threshold
        )

        return (
            flat_locs[nms_mask],
            flat_probs[nms_mask],
            class_indexes[nms_mask],
            image_indexes[nms_mask]
        )


    def locs_to_xyxy(self, locs, image_heights, image_widths):
        image_heights = image_heights.reshape(-1, 1, 1, 1)
        image_widths = image_widths.reshape(-1, 1, 1, 1)

        image_wh = torch.cat([image_widths, image_heights], dim=-1)

        xy = locs[:, :, :, 0:2] * image_wh
        wh = (locs[:, :, :, 2:4] - locs[:, :, :, 0:2]) * image_wh # surely this could just be locs[:, :, :, 2:4] * image_wh and then return cat([xy, xy2])?

        return torch.cat([xy, xy + wh], dim=-1)


    def xyxy_to_xywh(self, xyxy):
        return torch.cat([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]], dim=-1)


def eval_coco(args):
    device = torch.device(args.device)

    model = SSD300(
        args.topk, args.detection_threshold, args.iou_threshold, args.precision, args.batch_dim, args.trt_path
    ).to(device).eval()

    dataloader = get_val_dataloader(args)
    inv_map = {v: k for k, v in dataloader.dataset.label_map.items()}

    coco_ground_truth = get_coco_ground_truth(args)

    results = None
    start = time.time()

    for nbatch, (X, img_id, img_size, _, _) in enumerate(dataloader):
        print('Inference batch: {}/{}'.format(nbatch, len(dataloader)), end='\r')
        with torch.no_grad():
            batch_dim = X.size(0)
            if args.precision == 'fp16':
                X = X.to(torch.float16)
            X = X.to(device)
            image_heights, image_widths = [i.to(device) for i in img_size]

            if batch_dim < args.batch_dim:
                num_pad = args.batch_dim - batch_dim
                X = torch.cat([X, X[-1].expand(num_pad, *X[-1].size())], dim=0)
                image_heights = torch.cat([image_heights, image_heights[-1].repeat(num_pad)], dim=0)
                image_widths = torch.cat([image_widths, image_widths[-1].repeat(num_pad)], dim=0)

            bboxes, probs, class_indexes, image_indexes = model.forward_coco(X, image_heights, image_widths)

            # filter out pad results
            small_batch_filter = image_indexes < batch_dim
            bboxes = bboxes[small_batch_filter]
            probs = probs[small_batch_filter]
            class_indexes = class_indexes[small_batch_filter]
            image_indexes = image_indexes[small_batch_filter]

            mapped_labels = class_indexes.to('cpu')
            mapped_labels.apply_(lambda i: inv_map[i])
            image_ids = img_id[image_indexes.to('cpu')]

            batch_results = torch.cat([
                image_ids.cpu().unsqueeze(-1),
                bboxes.cpu(),
                probs.cpu().unsqueeze(-1),
                mapped_labels.unsqueeze(-1)
            ], dim=1)

            if results is not None:
                results = torch.cat([results, batch_results], dim=0)
            else:
                results = batch_results

    print()
    print(f'DONE (t={time.time() - start:.2f}).')

    results = results.numpy().astype(np.float32)

    coco_detections = coco_ground_truth.loadRes(results)

    E = COCOeval(coco_ground_truth, coco_detections, iouType='bbox')
    E.evaluate()
    E.accumulate()
    stdout = sys.stdout
    try:
        if args.output_path:
            sys.stdout = open(args.output_path, 'w')
        E.summarize()
    finally:
        if args.output_path:
            sys.stdout.close()
        sys.stdout = stdout
    print('mAP: {:.5f}'.format(E.stats[0]))


def export_engine(args):
    onnx_module = build_onnx(args)
    build_trt_engine(onnx_module, args)


def build_onnx(args):
    device = torch.device('cpu')
    val_dataloader = get_val_dataloader(args)

    for nbatch, (X, img_id, img_size, _, _) in enumerate(val_dataloader):
        inputs = X, img_size[0], img_size[1]
        break

    model = SSD300(args.topk, args.detection_threshold, args.iou_threshold, 'fp32', args.batch_dim, None, onnx_export=True).to(device).eval()

    onnx_buf = io.BytesIO()
    torch.onnx.export(
        model,
        inputs,
        onnx_buf,
        input_names=('tensor_nchw', 'image_heights', 'image_widths'),
        output_names=('bboxes', 'probs'),
        opset_version=11,
        export_params=True
    )
    onnx_buf.seek(0)
    onnx_module = shape_inference.infer_shapes(onnx.load(onnx_buf))

    onnx_path = os.path.splitext(args.trt_path)[0] + '.onnx'
    print('saving ONNX model to', onnx_path)
    onnx.save(onnx_module, onnx_path)
    return onnx_module



def build_trt_engine(onnx_module, args):
    logger = trt.Logger()

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(logger) as builder, builder.create_network(network_flags) as network, trt.OnnxParser(network, logger) as parser:
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 ** 31)

        # Precision setting
        if args.precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        elif args.precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = Int8Calibrator(args)

        print('parsing ONNX...')
        onnx_buf = io.BytesIO()
        onnx.save(onnx_module, onnx_buf)
        onnx_buf.seek(0)
        if not parser.parse(onnx_buf.read()):
            print(parser.num_errors, 'parser errors:')
            for i in range(parser.num_errors):
                print(parser.get_error(i))

        bboxes = network.get_output(0)  # Bounding boxes
        scores = network.get_output(1)  # Class confidence scores
        maxOutput = network.add_constant([], np.int32(args.topk).reshape(-1)).get_output(0)
        scoreThreshold = network.add_constant([], np.float32(args.detection_threshold).reshape(-1)).get_output(0)
        iouThreshold = network.add_constant([], np.float32(args.iou_threshold).reshape(-1)).get_output(0)

        network.unmark_output(bboxes)
        network.unmark_output(scores)

        print('Add INMS layers...')
        inms_layer = network.add_nms(
            bboxes,
            scores,
            maxOutput
        )
        inms_layer.topk_box_limit = args.topk

        inms_layer.set_input(3, iouThreshold)
        inms_layer.set_input(4, scoreThreshold)

        inms_layer.get_output(0).name = "SelectedIndices"
        inms_layer.get_output(1).name = "NumOutputBoxes"

        network.mark_output(inms_layer.get_output(0))
        network.mark_output(inms_layer.get_output(1))
        
        network.mark_output(bboxes)
        network.mark_output(scores)

        print('building CUDA engine...')
        engine = builder.build_serialized_network(network, config)
        if engine:
            print('saving CUDA engine to', args.trt_path)
            with open(args.trt_path, 'wb') as mf:
                mf.write(engine)

        return engine



def benchmark(args):
    app_start = time.time()

    prewarm_iters = 50
    bench_secs = 10

    val_dataloader = get_val_dataloader(args)

    for nbatch, (tensor_nchw, img_id, (image_heights, image_widths), _, _) in enumerate(val_dataloader):
        tensor_nchw, image_heights, image_widths = [t.to('cuda') for t in (tensor_nchw, image_heights, image_widths)]
        break

    batch_dim = tensor_nchw.size(0)

    update_fps, plot_thread = gpuplot.bg_plot(
        num_gpus=args.num_devices,
        sample_hz=5,
    )

    max_times = 10
    batch_times = []
    last_update = time.time()
    update_period = 0.5

    if args.runtime == 'pytorch':
        print(f'Runtime: Pytorch\nPrecision: {args.precision}\nBatch-dim: {args.batch_dim}\nTop-k: {args.topk}')
        model = SSD300(args.topk, args.detection_threshold, args.iou_threshold, args.precision, args.batch_dim, args.trt_path)
        model = model.eval().to('cuda')

        if args.precision == 'fp16':
            tensor_nchw, image_heights, image_widths = [t.to(torch.float16) for t in (tensor_nchw, image_heights, image_widths)]

        plot_thread.start()

        print('Prewarming model')
        for i in range(prewarm_iters):
            model(tensor_nchw, image_heights, image_widths)
            batch_times = (batch_times + [time.time()])[-max_times:]

        print(f'Beginning benchmark (+{time.time() - app_start:.1f})...')
        start_time = time.time()

        bench_iters = 0
        while True:
            model(tensor_nchw, image_heights, image_widths)
            batch_times = (batch_times + [time.time()])[-max_times:]
            if batch_times[-1] > last_update + update_period and len(batch_times) > 1:
                last_update = batch_times[-1]
                update_fps(args.batch_dim * (len(batch_times) - 1) / (batch_times[-1] - batch_times[0]))
            bench_iters += 1
            if time.time() > start_time + bench_secs:
                break

    elif args.runtime == 'trt':
        print(f'Runtime: TensorRT\nPrecision: {args.precision}\nBatch-dim: {args.batch_dim}\nTop-k: {args.topk}')
        np_to_torch_type = {
            np.float32: torch.float32,
            np.float16: torch.float16,
            np.int32: torch.int32,
            np.int64: torch.int64,
        }

        devices = [cuda.Device(i) for i in range(args.num_devices)]
        contexts = [devices[i].make_context() for i in range(args.num_devices)]

        for d in devices:
            pycuda.autoinit.context.pop()

        context_detail = []

        for device_id, context in enumerate(contexts):
            context.push()
            try:
                torch_device = torch.device('cuda', device_id)
                streams = [cuda.Stream() for i in range(args.num_streams_per_device)]

                tensors = {
                    name: t.clone().to(torch_device)
                    for name, t in [
                        ('tensor_nchw', tensor_nchw),
                        ('image_heights', image_heights),
                        ('image_widths', image_widths)
                    ]
                }

                model = SSD300(args.topk, args.detection_threshold, args.iou_threshold, args.precision, args.batch_dim, args.trt_path)

                trt_outputs, bindings = [[] for i in range(args.num_streams_per_device)], [[] for i in range(args.num_streams_per_device)]

                for stream_id in range(args.num_streams_per_device):
                    for binding_name in model.trt_engine:
                        shape = model.trt_engine.get_tensor_shape(binding_name)
                        shape = [tensor_nchw.size(0) if dim == -1 else dim for dim in shape]
                        dtype = trt.nptype(model.trt_engine.get_tensor_dtype(binding_name))
                        torch_type = np_to_torch_type[dtype]

                        if model.trt_engine.get_tensor_mode(binding_name) and (binding_name != "SelectedIndices") and (binding_name != "NumOutputBoxes") and (binding_name != "bboxes") and (binding_name != "probs"):
                            torch_input = tensors[binding_name].to(torch_type)
                            bindings[stream_id].append(int(torch_input.data_ptr()))
                        else:
                            torch_output = torch.zeros(tuple(shape), dtype=torch_type, device=torch_device)
                            trt_outputs[stream_id].append(torch_output)
                            bindings[stream_id].append(int(torch_output.data_ptr()))

                context_detail.append({
                    'streams': streams,
                    'model': model,
                    'trt_outputs': trt_outputs,
                    'bindings': bindings
                })

            finally:
                context.pop()

        event_queue = queue.Queue(args.num_devices * args.num_streams_per_device)

        def sync_streams(update_fps, batch_times, max_times, last_update, update_period):
            while True:
                ce = event_queue.get()
                if ce is None:
                    break
                else:
                    context, e = ce
                    context.push()
                    e.synchronize()
                    context.pop()

                    batch_times = (batch_times + [time.time()])[-max_times:]
                    if batch_times[-1] > last_update + update_period and len(batch_times) > 1:
                        last_update = batch_times[-1]
                        update_fps(args.batch_dim * (len(batch_times) - 1) / (batch_times[-1] - batch_times[0]))

        sync_thread = threading.Thread(target=sync_streams, args=(update_fps, batch_times, max_times, last_update, update_period))
        sync_thread.start()

        plot_thread.start()

        # for benchmarking purposes, just run model repeatedly on initial batch of inputs
        bench_iters = 0
        while True:
            if bench_iters == 0:
                print('Prewarming model')
            elif bench_iters == prewarm_iters:
                print(f'Beginning benchmark (+{time.time() - app_start:.1f})...')
                start_time = time.time()
            elif bench_iters > prewarm_iters and time.time() > start_time + bench_secs:
                break

            context_id = bench_iters % len(context_detail)
            context = contexts[context_id]
            context.push()
            try:
                detail = context_detail[context_id]
                stream_id = (bench_iters - context_id) % len(detail['streams'])
                stream = detail['streams'][stream_id]
                for i in range(detail['model'].trt_engine.num_io_tensors):
                    detail['model'].trt_context.set_tensor_address(detail['model'].trt_engine.get_tensor_name(i), detail['bindings'][stream_id][i])
                
                detail['model'].trt_context.execute_async_v3(
                    stream_handle=stream.handle)
                event = cuda.Event(cuda.event_flags.DISABLE_TIMING)
                event_queue.put((context, event.record(stream)))
            finally:
                context.pop()

            bench_iters += 1

        event_queue.put(None)
        while not event_queue.empty():
            pass
        bench_iters -= prewarm_iters

    total_time = time.time() - start_time

    update_fps(None)
    plot_thread.join()

    print(f'{bench_iters} batches, {bench_iters * batch_dim} images, {total_time:.2f} seconds total')
    print(f'{1000 * total_time / (bench_iters * batch_dim):.1f} ms per image')
    print(f'{(bench_iters * batch_dim) / total_time:.1f} FPS')

    if args.output_path:
        with open(args.output_path, 'w') as fout:
            print(f'{bench_iters} batches, {bench_iters * batch_dim} images, {total_time:.2f} seconds total', file=fout)
            print(f'{1000 * total_time / (bench_iters * batch_dim):.1f} ms per image', file=fout)
            print(f'{(bench_iters * batch_dim) / total_time:.1f} FPS', file=fout)



def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['coco', 'export', 'bench'], default='coco')
    p.add_argument('--runtime', choices=['pytorch', 'trt'], default='pytorch')
    p.add_argument('--output-path')

    p.add_argument('--device', default=('cuda:0' if torch.cuda.is_available() else 'cpu'))
    p.add_argument('--detection-threshold', default=0.05, type=float)
    p.add_argument('--iou-threshold', default=0.5, type=float)
    p.add_argument('--topk', default=256, type=int)
    p.add_argument('--batch-dim', default=16, type=int)
    p.add_argument('--precision', default='fp16')

    p.add_argument('--num-streams-per-device', type=int, default=4)
    p.add_argument('--num-devices', type=int, default=1)

    p.add_argument('--eval-batch-size', default=None)
    p.add_argument('--data', default='/data/coco2017')
    p.add_argument('--num-workers', default=2)

    args = p.parse_args()
    args.eval_batch_size = args.batch_dim

    if args.runtime == 'trt':
        args.trt_path = f'models/ssd300.{args.precision}.b{args.batch_dim}.k{args.topk}.plan'
    else:
        args.trt_path = None

    if args.mode =='coco' and args.precision == 'int8' and args.runtime != 'trt':
        print('incompatible args')
        sys.exit(1)

    return args



if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'export':
        export_engine(args)
    elif args.mode == 'coco':
        eval_coco(args)
    elif args.mode == 'bench':
        benchmark(args)




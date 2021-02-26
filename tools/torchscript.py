import argparse
from functools import partial

import mmcv
import numpy as np
import torch
import torch._C
import torch.serialization
from mmcv.runner import load_checkpoint
from torch import nn

from mmseg.models import build_segmentor

torch.manual_seed(3)


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output

def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape) # 真实测试的时候可以换成真实的图片处理后的结果
    segs = rng.randint(
        low=0, high=num_classes - 1, size=(N, 1, H, W)).astype(np.uint8)
    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
    } for _ in range(N)]
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_semantic_seg': torch.LongTensor(segs)
    }
    return mm_inputs

def torchscript(model, input_shape, output_file='tmp.pt', verify=False):
    """Export Pytorch model to torchscript model and verify the outputs are same
    between Pytorch and torchscript.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and torchscript.
            Default: False.
    """
    model.cpu().eval()
    if isinstance(model.decode_head, nn.ModuleList):
        num_classes = model.decode_head[-1].num_classes
    else:
        num_classes = model.decode_head.num_classes
    
    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    # img_list = [img[None, :] for img in imgs]
    img_list = imgs
    img_meta_list = [[img_meta] for img_meta in img_metas]

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(model.forward, img_metas=img_meta_list, return_loss=False)

    model_script = torch.jit.trace(model, (img_list,))
    model_script.save(output_file)

    model.forward = origin_forward
    if verify:
        sm = torch.jit.load(output_file)

        # pytorch output
        pytorch_result = model(img_list, img_meta_list, return_loss=False)
        pytorch_result = pytorch_result.detach().numpy()

        # torchscript output
        sm_result = sm(img_list)
        sm_result = sm_result.detach().numpy()

        if not np.allclose(pytorch_result, sm_result):
            raise ValueError(
                'The outputs are different between Pytorch and TorchScript')
        print('The outputs are same between Pytorch and TorchScript')




def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMSeg to torchscript')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument(
        '--verify', action='store_true', help='verify the onnx model')
    parser.add_argument('--output-file', type=str, default='tmp.pt')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[256, 256],
        help='input image size')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    segmentor = build_segmentor(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg')
    )
    # convert SyncBN to BN
    segmentor = _convert_batchnorm(segmentor)

    if args.checkpoint:
        load_checkpoint(segmentor, args.checkpoint, map_location='cpu')

    # conver model to torchscript
    torchscript(
        segmentor,
        input_shape,
        output_file=args.output_file,
        verify=args.verify
    )



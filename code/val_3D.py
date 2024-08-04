import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt, random_state=0):

    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and np.unique(gt).size > 1:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def calculate_bs_metric_percase(pred, gt, num_bootstraps=1000, seed=0):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    num_slices = pred.shape[0]
    bs_results = []
    # calculate TP, FP, FN for each slice
    tp_arr = np.zeros(num_slices)
    fp_arr = np.zeros(num_slices)
    fn_arr = np.zeros(num_slices)
    for i in range(num_slices):
        tp = np.sum(pred[i] * gt[i])
        fp = np.sum(pred[i] * (1 - gt[i]))
        fn = np.sum((1 - pred[i]) * gt[i])
        tp_arr[i] = tp
        fp_arr[i] = fp
        fn_arr[i] = fn
    for i in range(num_bootstraps):
        bs_hd95 = 0.0
        current_seed = seed + i
        rng = np.random.RandomState(current_seed)
        sample_slices = rng.choice(np.arange(num_slices), size=num_slices, replace=True)
        bs_tp, bs_fp, bs_fn = np.sum(tp_arr[sample_slices]), np.sum(fp_arr[sample_slices]), np.sum(fn_arr[sample_slices])
        bs_dice_numer = 2 * bs_tp
        bs_dice_denom = bs_dice_numer + bs_fp + bs_fn
        if bs_dice_denom > 0:
            bs_dice = bs_dice_numer / bs_dice_denom
            bs_results.append([bs_dice, bs_hd95])
        else:
            bs_results.append([0, 0])
    bs_results = np.array(bs_results)
    bs_dice_results = bs_results[:, 0]
    bs_hd95_results = bs_results[:, 1]
    return bs_dice_results, bs_hd95_results


def test_single_volume(image, label, net, classes, patch_size=[16, 256, 256], in_chns=1, num_bootstraps=None, seed=0,
                       gpus="cuda:0"):
    print("image.shape: ", image.shape)
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        z, x, y = image.shape[0], image.shape[1], image.shape[2]
        image = zoom(
            image, (patch_size[0] / z, patch_size[1] / x, patch_size[2] / y), order=0)
        input = torch.from_numpy(image)
        if in_chns == 3:
            input = torch.stack([input] * 3, dim=0)
            input = input.unsqueeze(0).float().to(gpus)
        else:
            input = input.unsqueeze(0).unsqueeze(0).float().to(gpus)
        print("after transform, image.shape: ", image.shape)
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction = zoom(out, (patch_size[0] / z, patch_size[1] / x, patch_size[2] / y), order=0)
            label = zoom(label, (patch_size[0] / z, patch_size[1] / x, patch_size[2] / y), order=0)
    else:
        raise ValueError("Only image.shape==3 is supported for now")

    print("prediction.shape: ", prediction.shape)
    print("label.shape: ", label.shape)

    metric_list = []
    for i in range(1, classes):
        if isinstance(num_bootstraps, int):
            bs_dice, bs_hd95 = calculate_bs_metric_percase(prediction == i, label == i, num_bootstraps, seed)
            metric_list.append([bs_dice, bs_hd95])
        else:
            metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

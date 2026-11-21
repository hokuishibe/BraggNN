import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import BraggNNDataset
import torchao.quantization.pt2e


def make_gaussian():
    X_test = np.zeros((11, 11))
    x_cen, y_cen = 5.5, 5.5

    sig_x, sig_y = 2.0, 2.0
    for x in range(11):
        for y in range(11):
            X_test[y][x] = 1000*(np.exp(-(x-x_cen)*(x-x_cen)/2*sig_x - (y-y_cen)*(y-y_cen)/2*sig_y))

    X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
    return torch.from_numpy(X_test[np.newaxis, np.newaxis].astype('float32'))


def generate_dummy_input():
    psz = 11  # BRAGGNN_PATCH_SIZE
    input_data = np.zeros(psz * psz, dtype=np.float32)

    center_x = 5.5
    center_y = 5.5
    sigma = 2.0

    for i in range(psz):
        for j in range(psz):
            dx = float(j) - center_x
            dy = float(i) - center_y
            dist_sq = dx * dx + dy * dy
            input_data[i * psz + j] = np.exp(-dist_sq / (2.0 * sigma * sigma))

    # Add some noise
    for i in range(psz * psz):
        input_data[i] += (i % 7) / 70.0

    # Normalize to [0, 1]
    min_val = input_data.min()
    max_val = input_data.max()

    if max_val > min_val:
        input_data = (input_data - min_val) / (max_val - min_val)

    print("Generated dummy input data (11x11 patch with Gaussian-like peak)")

    # Reshape to (1, 1, 11, 11) for model input
    return torch.from_numpy(input_data.reshape(1, 1, psz, psz))

def main():
    #model = BraggNN(imgsz=11, fcsz=(16, 8, 4, 2))
    #mdl_fn = 'models/fc16_8_4_2-sz11.pth'
    #model.load_state_dict(torch.load(mdl_fn, map_location=torch.device('cpu')))
    mdl_fn = 'models/int8_16_8_4_2-sz11.pth'
    mdl_fn2 = 'models/int8_16_8_4_2-sz11-wo-nlb.pth'
    model = torch.export.load(mdl_fn).module()
    model2 = torch.export.load(mdl_fn2).module()

    ds_input = BraggNNDataset(psz=11, rnd_shift=0, use='validation')
    dl_input = DataLoader(dataset=ds_input, batch_size=512, shuffle=False, \
                          num_workers=8, prefetch_factor=512, drop_last=False, pin_memory=True)

    # input_tensor = make_gaussian()

    preds, gts = [], []
    preds2, gts2 = [], []
    with torch.no_grad():
        for x, y in dl_input:
            for i in range(x.shape[0]):
                preds.append(model(x[i:i+1]).numpy())
                preds2.append(model2(x[i:i+1]).numpy())
            gts.append(y.numpy())
            gts2.append(y.numpy())
    pred = np.concatenate(preds)
    pred2 = np.concatenate(preds2)
    gt = np.concatenate(gts)
    gt2 = np.concatenate(gts2)
    err = np.abs(gt - pred) * 11
    err2 = np.abs(gt2 - pred2) * 11
    mae_x, mae_y = err.mean(axis=0)
    mae_x2, mae_y2 = err2.mean(axis=0)
    print('Original, samples: %d, MAE: (%.4f, %.4f)' % (len(err), mae_x, mae_y))
    print('/wo NLB, samples: %d, MAE: (%.4f, %.4f)' % (len(err2), mae_x2, mae_y2))


if __name__ == "__main__":
    main()

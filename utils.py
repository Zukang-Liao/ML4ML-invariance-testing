import numpy as np
import os
import argparse
import torchvision
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

CIFAR10_DIR = '/Users/z.liao/dataset/CIFAR10'


def get_even_array(a_min, a_max, nb_elements):
    assert nb_elements >=2, "Number of elements must be greater than 1"
    result = np.zeros(nb_elements)
    total = a_max - a_min
    increment = total/(nb_elements-1)
    for i in range(nb_elements):
        result[i] = a_min + i * increment
    return result

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/Users/z.liao/oxfordXAI/repo/XAffine/plots/test_results.npy")
    parser.add_argument("--data_path1", type=str, default="/Users/z.liao/oxfordXAI/repo/XAffine/plots/test_results_45_15.npy")
    parser.add_argument("--data_path2", type=str, default="/Users/z.liao/oxfordXAI/repo/XAffine/plots/test_results.npy")
    parser.add_argument("--data_path3", type=str, default="/Users/z.liao/oxfordXAI/repo/XAffine/plots/test_results_15_45.npy")
    parser.add_argument("--out_path", type=str, default="/Users/z.liao/oxfordXAI/repo/XAffine/plots/test_results_45_45.npy")
    parser.add_argument("--plot_foldername", type=str, default="1515")
    args = parser.parse_args()
    args.plot_dir = os.path.join(os.path.dirname(args.data_path), args.plot_foldername, "ori")
    return args

def merge_saved_results(args):
    data1 = np.load(args.data_path1, mmap_mode="r")
    data2 = np.load(args.data_path2, mmap_mode="r")
    data3 = np.load(args.data_path3, mmap_mode="r")
    data = np.concatenate([data1, data2, data3], axis=0)
    np.save(args.out_path, data)

def svd(incorr_f, corr_f):
    incorr_f, incorr_m = preprocess(incorr_f)
    corr_f, corr_m = preprocess(corr_f)
    incorr_U, incorr_s, incorr_V = np.linalg.svd(incorr_f, full_matrices=False)
    corr_U, corr_s, corr_V = np.linalg.svd(corr_f, full_matrices=False)
    results = [[incorr_U, incorr_s, incorr_V, incorr_m], [corr_U, corr_s, corr_V, corr_m]]
    return results

def verify_paths(args):
    if args.train:
        assert "train" in args.data_filename, "result path not for training set"
    else:
        assert "test" in args.data_filename, "result path not for testing set"

def verify_rotation(args, img, test_angles):

    def plot_mat(mat, title, plot_name):
        plt.pcolormesh(mat)
        plt.title(title)
        plt.colorbar()
        plt.savefig(os.path.join(args.plot_dir, plot_name))
        plt.clf()
        plt.close()

    nb_angles = len(test_angles)
    imgs = np.zeros((nb_angles,) + np.array(img).shape)
    results = np.zeros([nb_angles, nb_angles])
    zeros_map = np.zeros([nb_angles, nb_angles])
    for i, angle in enumerate(test_angles):
        imgs[i] = np.array(TF.rotate(img, angle))
    for i in range(nb_angles-1):
        zeros_map[i, i] = 10
        for j in range(i+1, nb_angles):
            pixel_diff = np.mean(np.abs(np.subtract(imgs[i], imgs[j])))
            results[i, j] = pixel_diff
            results[j, i] = pixel_diff
            if pixel_diff == 0:
                zeros_map[i, j] = 10
                zeros_map[j, i] = 10
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)
    plot_mat(results, "Rotation verification: pixel value difference", "rotation_verification.jpg")
    plot_mat(zeros_map, "Rotation verification: 10: the same picture", "rotation_verification_zeromap.jpg")


def get_relations(mat, l2_norm=True, flip_y=True):
    dim = [len(mat), mat[0].shape[0], mat[0].shape[0]]
    corrcoefs, cos_dists, l2_dists = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    for i, m in enumerate(mat):
        corrcoefs[i] = np.corrcoef(m)
        cos_dists[i] = cosine_distances(m)
        l2_dists[i] = euclidean_distances(m)
        if l2_norm:
            nb_examples = m.shape[1]
            l2_dists[i] /= np.sqrt(nb_examples)
    if flip_y:
        corrcoefs = np.flip(corrcoefs, axis=1)
        cos_dists = np.flip(cos_dists, axis=1)
        l2_dists = np.flip(l2_dists, axis=1)
    return corrcoefs, cos_dists, l2_dists

def merge_relations(relations, nb_angles):
    gap = 2
    h, w = nb_angles + 2 * gap, nb_angles * 2 + 3 * gap
    result = np.zeros([len(relations), h, w])
    for j, relation in enumerate(relations):
        for i, r in enumerate(relation):
            result[j][gap:gap+nb_angles, (i+1)*gap+i*nb_angles:(i+1)*(gap+nb_angles)] = r
    return result

# This method works only for CIFAR dataset with 32,32 dim --> -1, 0, 1 degree rotations are the same
def fillin_relation_diagonal(mat, flip_y=True):
    dim = mat.shape[0]
    idx0 = int(dim/2)
    centre = list(range(idx0-1, idx0+2))
    if flip_y:
        for i, row in enumerate(mat):
            if i == 0:
                row[-1] = row[-2]
            elif i == dim-1:
                row[0] = row[1]
            elif i in centre:
                row[centre] = (mat[centre[0], centre[-1]+1] + mat[centre[-1]+1, centre[-1]]) / 2
            else:
                row[dim-1-i] = (row[dim-i]+mat[i+1, dim-1-i]) / 2
                # row[dim-1-i] = (row[dim-i]+mat[i+1, dim-1-i]+mat[i+1, dim-i])/ 3
    else:
        for i, row in enumerate(mat):
            if i == 0:
                row[i] = row[i+1]
            elif i == dim-1:
                row[i] = row[i-1]
            elif i in centre:
                row[centre] = (mat[centre[0], centre[0]-1] + mat[centre[-1]+1, centre[0]]) / 2
            else:
                row[i] = (row[i-1]+mat[i+1, i]) / 2
                # row[i] = (row[i-1]+mat[i+1, i]+mat[i+1, i-1]) / 3
    return mat

def get_continuity(imgs, flip_y=True):
    def get_diagonal_std(img, i, forward):
    # return the i-th "diagonal" std starting from the i-th row
        values = []
        j = 0
        while i>=0 and i<len(img):
            values.append(img[i, j])
            if forward:
                i -= 1
            else:
                i += 1
            j += 1
        return np.std(values), len(values)
    results = []
    for img in imgs:
        img = fillin_relation_diagonal(img.copy(), flip_y)
        d_std = []
        dim = img.shape[0]
        m = np.sum(img) / (dim*dim-dim)
        for i in range(1, dim):
            std, w = get_diagonal_std(img, i, forward=flip_y)
            # d_std.append(std)
            d_std.append(w*std/dim)
        # results.append(np.mean(d_std))
        results.append(np.sum(d_std)/m)
    return results

def get_asymmetry(imgs, flip_y=True):
    # Symmetric about the second diagonal
    results = []
    for img in imgs:
        dim = img.shape[0]
        m = np.sum(img) / (dim*dim-dim) # exclude the primary diagonal
        values = []
        if flip_y:
            for i in range(dim):
                for j in range(i):
                    values.append(np.abs(img[i, j]-img[j, i]))
        else:
            for i in range(dim):
                for j in range(dim-i):
                    values.append(np.abs(img[i, j]-img[dim-1-j, dim-1-i]))
        results.append(np.mean(values)/m)
    return results


def log_asymm_ctny(txt_path, train, metric_name, ctny, asymm, grad_scs, class_id, test_angles, nb_examples, feature_name=None):
    if train:
        txt_begin = "Train"
    else:
        txt_begin = "Test"
    txt_begin += f", Test angle: [{test_angles[0]}: {test_angles[-1]}]\n"
    if feature_name is not None:
        txt_begin += f"{feature_name} Mean"
    else:
        txt_begin += f"confidence score"
    txt_begin += f", {metric_name}"
    if class_id is not None:
        txt_begin += f", class_{class_id}"
    if len(nb_examples) == 1:
        txt_begin += f"\nNumber of Correct examples:{nb_examples[0]}\n"
    else:
        txt_begin += f"\nIncorrect examples: {nb_examples[0]}, Correct examples:{nb_examples[1]}\n"
    txt_begin += "---------------------------------------------------\n"


    with open(txt_path, "a") as txtfile:
        txtfile.write(txt_begin)
        if len(ctny) == 1:
            txtfile.write("Correct example\n")
            txtfile.write(f"Discontinuity: {ctny[0]:.5f}\n")
            txtfile.write(f"Asymmetry: {asymm[0]:.5f}\n")
            for gd in grad_scs:
                txtfile.write(f"Gradient {gd}:\n")
                grad_sc = grad_scs[gd]
                txtfile.write(f"    Mean: {grad_sc[0,0]:.5f}\n")
                txtfile.write(f"    Std: {grad_sc[0,1]:.5f}\n")
                txtfile.write(f"    Wstd: {grad_sc[0,2]:.5f}\n")
            txtfile.write("\n\n")
        else:
            txtfile.write("Discontinuity:\n")
            txtfile.write(f"    Incor: {ctny[0]:.5f}, Corr: {ctny[1]:.5f}\n")
            txtfile.write("Asymmetry:\n")
            txtfile.write(f"    Incor: {asymm[0]:.5f}, Corr: {asymm[1]:.5f}\n")
            for gd in grad_scs:
                txtfile.write(f"Gradient {gd}:\n")
                grad_sc = grad_scs[gd]
                txtfile.write(f"    Incor_mean: {grad_sc[0,0]:.5f}, Corr_mean: {grad_sc[1,0]:.5f}\n")
                txtfile.write(f"    Incor_std: {grad_sc[0,1]:.5f}, Corr_std: {grad_sc[1,1]:.5f}\n")
                txtfile.write(f"    Incor_wstd: {grad_sc[0,2]:.5f}, Corr_wstd: {grad_sc[1,2]:.5f}\n")
            txtfile.write("\n\n")


if __name__ == "__main__":
    args = argparser()
    print(args)
    data = torchvision.datasets.CIFAR10(CIFAR10_DIR, train=False, download=True)
    img, label = data[0]
    verify_rotation(args, img, test_angles=list(range(-15,15+1)))
    # merge_saved_results(args)
import cv2
import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset.kitti import Kitti_Dataset
from model.usnet import USNet
from utils import fast_hist, getScores


def test(args, model, dataloader):
    print_prediction_image = False

    print('start test!')
    with torch.no_grad():
        model.eval()
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, sample in enumerate(dataloader):
            image, depth, label = sample['image'], sample['depth'], sample['label']
            oriHeight, oriWidth = sample['oriHeight'], sample['oriWidth']
            oriWidth = oriWidth.cpu().numpy()[0]
            oriHeight = oriHeight.cpu().numpy()[0]

            if torch.cuda.is_available() and args.use_gpu:
                image = image.cuda()
                depth = depth.cuda()
                label = label.cuda()

            evidence, evidence_a, alpha, alpha_a = model(image, depth)

            s = torch.sum(alpha_a, dim=1, keepdim=True)
            p = alpha_a / (s.expand(alpha_a.shape))

            pred = p[:,1]
            pred = pred.view(args.crop_height, args.crop_width)
            pred = pred.detach().cpu().numpy()
            pred = np.array(pred)

            # save predict image
            visualize = cv2.resize(pred, (oriWidth, oriHeight))
            visualize = np.floor(255*(visualize - visualize.min()) / (visualize.max()-visualize.min()))
            img_path = sample['img_path'][0]
            img_name = img_path.split('/')[-1]
            #save_name = img_name.split('_')[0] + '_road_' + img_name.split('_')[1]
            save_name = img_name.split('_')[0]+'_road_'+img_name.split('_')[1]+'_'+img_name.split('_')[2]
            print(os.path.join(args.save_path, save_name))

            # get predict image
            if print_prediction_image:
                print(sample['img_path'])
                img_show = cv2.imread(sample['img_path'][0], 0)
                cv2.imshow('image', img_show)
                cv2.waitKey(0)

            img_o = cv2.imread(sample['img_path'][0], 0)
            img_p = np.uint8(visualize)

#region path planning

            img_p2 = img_p.copy()

            img_p2_height, img_p2_width = img_p2.shape
            img_p2_height_cutoff = img_p2_height // 20

            img_p2_slice_height_offset = img_p2_height_cutoff

            for x in range(10):
                img_p2_slice_start = img_p2_height - img_p2_slice_height_offset + img_p2_height_cutoff
                img_p2_slice_end = img_p2_height - img_p2_slice_height_offset

                img_p2_slice = img_p2[img_p2_slice_end : img_p2_slice_start,:]

                centroid = np.mean(np.argwhere(img_p2_slice), axis=0)

                if np.isnan(centroid).any():
                    continue
                else:
                    centroid_x, centroid_y = int(centroid[1]), int(centroid[0])
                    img_p2 = cv2.circle(img_p2, (centroid_x, img_p2_height - img_p2_slice_height_offset + img_p2_height_cutoff - centroid_y), radius=0, color=(0, 0, 0), thickness=2)
                    img_p2 = cv2.circle(img_p2, (img_p2_width // 2, img_p2_height - img_p2_slice_height_offset + img_p2_height_cutoff - centroid_y),radius=1, color=(0, 0, 0), thickness=2)

                img_p2_slice_height_offset += img_p2_height_cutoff
#endregion
            #img_h = cv2.hconcat([img_o, img_p, img_p2])
            img_h = cv2.hconcat([img_o, img_p2])
            cv2.imwrite(os.path.join(args.save_path, save_name), img_h)
            # cv2.imwrite(os.path.join(args.save_path, save_name), np.uint8(visualize))

            pred = np.uint8(pred > 0.5)
            pred = cv2.resize(pred, (oriWidth, oriHeight), interpolation=cv2.INTER_NEAREST)

            # get label image
            label = label.squeeze()
            label = label.cpu().numpy()
            label = np.array(label)
            label = cv2.resize(np.uint8(label), (oriWidth, oriHeight), interpolation=cv2.INTER_NEAREST)

            hist += fast_hist(label.flatten(), pred.flatten(), args.num_classes)
        F_score, pre, recall, fpr, fnr = getScores(hist)
        print('F_score: %.3f' % F_score)
        print('pre : %.3f' % pre)
        print('recall: %.3f' % recall)
        print('fpr: %.3f' % fpr)
        print('fnr: %.3f' % fpr)
    return F_score, pre, recall, fpr, fnr


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the pretrained weights of model')
    parser.add_argument('--crop_height', type=int, default=384, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1248, help='Width of cropped/resized input image to network')
    parser.add_argument('--data', type=str, default='', help='Path of testing data')
    parser.add_argument('--dataset', type=str, default="KITTI", help='Dataset you are using.')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--backbone_name', type=str, default="resnet18", help='The backbone model you are using.')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--num_classes', type=int, default=2, help='num of object classes (with void)')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='Path to save predict image')
    args = parser.parse_args(params)

    dataset = Kitti_Dataset(args, root=args.data, split='testing')
    #dataset = Kitti_Dataset(args, root=args.data, split='validating')
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = USNet(args.num_classes, args.backbone_name)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load trained model
    print('load model from %s ...' % args.checkpoint_path)
    if torch.cuda.is_available() and args.use_gpu:
        model.load_state_dict(torch.load(args.checkpoint_path))
    else:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device('cpu')))
    print('Done!')

    # make save folder
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    test(args, model, dataloader)


if __name__ == '__main__':
    params = [
        '--checkpoint_path', './log/KITTI_model/usnet_best.pth',
        '--data', './data/KITTI',
        '--batch_size', '1',
        '--backbone_name', 'resnet18',
        '--cuda', '0',
        '--num_classes', '2',
        '--save_path', './result/kitti_test',
        '--use_gpu', 'False',
    ]
    main(params)

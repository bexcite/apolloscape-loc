import torch
from datasets.apolloscape import Apolloscape
from utils.common import draw_poses, calc_poses_params
from utils.common import draw_record, make_video
import numpy as np
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Plot Apolloscape dataset poses per record or generate video of a record")
    parser.add_argument("--data", metavar="DIR", required=True,
                        help="Path to Apollodataset")
    parser.add_argument("--road", metavar="ROAD_DIR", default="road03_seg",
                        help="Path to the road within ApolloScape")
    parser.add_argument("--show-records-count", dest="show_records_count", action="store_true",
                        help="Print all records with counts")
    parser.add_argument("--record", metavar="RECORD_DIR",
                        help="Path to the record directory within road")
    parser.add_argument("--sample-idx", metavar="SAMPLE_IDX", type=int,
                        help="Index to show. Should be in [0, len(dataset)] range")
    parser.add_argument("--output-dir", metavar="OUTPUT_DIR", default="output_data",
                        help="Path save figures and videos")
    parser.add_argument("--video", metavar="VIDEO_OUT_FILE", type=str, action="store", default=None, const="", nargs="?",
                        help="Generate and save video of an animated record to a file")
    parser.add_argument("--no-display", dest="no_display", action="store_true", default=False,
                        help="Don't show graphs on screen")


    return parser.parse_args()



def main():
    args = get_args()

    transform = transforms.Compose([transforms.Resize(250)])
    apollo_dataset = Apolloscape(root=os.path.join(args.data), road=args.road, transform=transform, record=args.record)

    print(apollo_dataset)

    if args.show_records_count:
        print("=== Records count: ")
        
        recs_num = apollo_dataset.get_records_counts()
            
        recs_num = sorted(recs_num.items(), key=lambda kv: kv[1], reverse=True)
        print("\n".join(["\t{} - {}".format(r[0], r[1]) for r in recs_num ]))
        return

    if args.video is not None:

        # Generate video for the record
        if len(args.video) > 1:
            video_outfile = os.path.join(os.path.expanduser(args.output_dir), args.video)
        else:
            # Make filename for the video
            video_output_path = os.path.join(os.path.expanduser(args.output_dir), "videos")
            video_outfile = os.path.join(video_output_path, "{}_{}.mp4".format(
                apollo_dataset.road, apollo_dataset.record))


        make_video(apollo_dataset, outfile=video_outfile)


    else:

        # No video, just draw images from the sample

        sample_idx = args.sample_idx
        if sample_idx is None:
            sample_idx = np.random.randint(len(apollo_dataset))

        fig = draw_record(apollo_dataset, idx=sample_idx)

        # Make output_dirs for graphs
        output_path = os.path.join(os.path.expanduser(args.output_dir))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Save figure
        image_fname = os.path.join(
            output_path, "{}_{}_{:05d}.png".format(
              apollo_dataset.road, apollo_dataset.record, sample_idx))
        fig.savefig(image_fname)

        # Show graph
        if not args.no_display:
            plt.show()

        plt.close(fig)


if __name__=="__main__":
    main()

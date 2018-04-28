python ./tools/extract_tsn_feature.py \
    /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_frames/bbt_frames  \
    --modality rgb \
    --h5_path_feat /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_feature/bbt_rgb_k400_feat.h5  \
    --h5_path_score /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_feature/bbt_rgb_k400_score.h5

python ./tools/extract_tsn_feature.py \
    /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_frames/castle_frames  \
    --modality rgb \
    --h5_path_feat /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_feature/castle_rgb_k400_feat.h5  \
    --h5_path_score /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_feature/castle_rgb_k400_score.h5

python ./tools/extract_tsn_feature.py \
    /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_frames/friends_frames  \
    --modality rgb \
    --h5_path_feat /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_feature/friends_rgb_k400_feat.h5  \
    --h5_path_score /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_feature/friends_rgb_k400_score.h5

python ./tools/extract_tsn_feature.py \
    /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_frames/grey_frames  \
    --modality rgb \
    --h5_path_feat /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_feature/grey_rgb_k400_feat.h5  \
    --h5_path_score /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_feature/grey_rgb_k400_score.h5

python ./tools/extract_tsn_feature.py \
    /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_frames/house_frames  \
    --modality rgb \
    --h5_path_feat /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_feature/house_rgb_k400_feat.h5  \
    --h5_path_score /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_feature/house_rgb_k400_score.h5

python ./tools/extract_tsn_feature.py \
    /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_frames/met_frames  \
    --modality rgb \
    --h5_path_feat /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_feature/met_rgb_k400_feat.h5  \
    --h5_path_score /net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/video_feature/met_rgb_k400_score.h5

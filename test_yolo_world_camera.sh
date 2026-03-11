# Override with env: CONF_THRES=0.40 ./test_yolo_world_camera.sh
CONF_THRES="${CONF_THRES:-0.50}"

# Optional (only when auto-detect does not work with your HEF):
#   --input-image-name <image_input_name> \
#   --input-text-name <text_input_name> \
CLIP_HEF=models/clip_vit_b_32_text_encoder.hef
.venv/bin/python scripts/test_yolo_world_camera.py \
  --text "bottle" \
  --prompt-template "a photo of {}" \
  --clip-hef "${CLIP_HEF}" \
  --hef models/yolo_world_v2s.hef \
  --camera-index 0 \
  --libcamerify on \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 60 \
  --conf-thres "${CONF_THRES}" \
  --iou-thres 0.45 \
  --show

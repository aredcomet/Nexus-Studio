echo "PROMPT"
cat prompt.txt
echo "------------------------------------------"
echo "model: gemma-3-270m:"
python -m mlx_lm generate \
    --model ./models/gemma-3-270m \
    --max-tokens 100 \
    --prompt "$(cat prompt.txt)"
echo "------------------------------------------"
echo "model: gemma-3-270m-270m:"
python t5v2/generate.py \
  --config models/t5gemma-2-270m-270m/config.json \
  --weights weights/t5gemma-2-270m-270m/weights.safetensors \
  --processor models/t5gemma-2-270m-270m \
  --prompt "$(cat prompt.txt)" \
  --max-tokens 100

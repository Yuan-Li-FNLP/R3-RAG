for i in {0..9999}; do
    cp /remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/split/hotpotqa/train/$i.jsonl /remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/split/hotpotqa/train_3w/
    echo "Copied $i.jsonl"
done
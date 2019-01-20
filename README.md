# Kaggle Human Protein Atlas Image Classification

In this competition, Kagglers will develop models capable of classifying mixed patterns of proteins in microscope images. The Human Protein Atlas will use these models to build a tool integrated with their smart-microscopy system to identify a protein's location(s) from a high-throughput image.

Proteins are “the doers” in the human cell, executing many functions that together enable life. Historically, classification of proteins has been limited to single patterns in one or a few cell types, but in order to fully understand the complexity of the human cell, models must classify mixed patterns across a range of different human cells.

Images visualizing proteins in cells are commonly used for biomedical research, and these cells could hold the key for the next breakthrough in medicine. However, thanks to advances in high-throughput microscopy, these images are generated at a far greater pace than what can be manually evaluated. Therefore, the need is greater than ever for automating biomedical image analysis to accelerate the understanding of human cells and disease.

### Protocol:

- Dataset: Original + hpa
- Image  : 512x512
- Color  : [G, (R+Y)/2, (B+Y)/2]
- Model  : resnet18; resnet50; se_resnet
- TTA    : ON
- Ensamble: ON

### Results:

| Branch   | Name     | LB Pub     | LB Pri    | Description               |
|---------:|---------:|:----------:|:---------:|:--------------------------|
| master   |cIN_(oo_) | 0.58       | 0.52      | Top 6% 112/2172           |


See: https://www.kaggle.com/c/human-protein-atlas-image-classification/leaderboard


## Dataset

See: https://www.kaggle.com/c/human-protein-atlas-image-classification/data

You will need to download a copy of the images. Due to size, we have provided two versions of the same images. On the data page below, you will find a scaled set of 512x512 PNG files in train.zip and test.zip. Alternatively, if you wish to work with full size original images (a mix of 2048x2048 and 3072x3072 TIFF files) you may download train_full_size.7z and test_full_size.7z from here (warning: these are ~250 GB total).

Donwload:

    kaggle competitions download -c human-protein-atlas-image-classification


## Visualize result with Visdom

We now support Visdom for real-time loss visualization during training!

To use Visdom in the browser:

    # First install Python server and client 
    pip install visdom
    # Start the server (probably in a screen or tmux)
    python -m visdom.server -env_path runs/visdom/
    # http://localhost:8097/

## Visualize jupyter notebook

    jupyter notebook --port 8080 --allow-root --ip 0.0.0.0 --no-browser


# Links 


- https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
- https://github.com/spytensor/kaggle_human_protein_baseline
- https://github.com/raytroop/human-protein-atlas-image-classification
- https://github.com/Cadene/pretrained-models.pytorch


# Credits

- Pedro D. Marrero Fernandez 
- Heitor Rapela Medeiros

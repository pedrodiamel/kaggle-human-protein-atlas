import xml.etree.ElementTree as etree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from skimage import io
from tqdm import tqdm
from collections import defaultdict
import os
import warnings
warnings.filterwarnings("ignore")
print(os.listdir('.'))

PROTEINATLAS_XML_PATH = "proteinatlas.xml"
TRAIN_EXTRA_PATH = "train_extra.csv"




# There are more labels in the extra training data than there are in the official training data
name_to_label_dict = {'nucleoplasm': 0, 'nuclear membrane': 1, 'nucleoli': 2, 'nucleoli fibrillar center': 3,
                      'nuclear speckles': 4, 'nuclear bodies': 5, 'endoplasmic reticulum': 6, 'golgi apparatus': 7,
                      'peroxisomes': 8, 'endosomes': 9, 'lysosomes': 10, 'intermediate filaments': 11,
                      'actin filaments': 12, 'focal adhesion sites': 13, 'microtubules': 14, 'microtubule ends': 15,
                      'cytokinetic bridge': 16, 'mitotic spindle': 17, 'microtubule organizing center': 18,
                      'centrosome': 19, 'lipid droplets': 20, 'plasma membrane': 21, 'cell junctions': 22,
                      'mitochondria': 23, 'aggresome': 24, 'cytosol': 25, 'cytoplasmic bodies': 26,
                      'rods & rings': 27, 'midbody': [16, 12], 'midbody ring': [16, 12], 'cleavage furrow': 16, 'vesicles': [8, 9, 10, 20]}



df = None
if((TRAIN_EXTRA_PATH in os.listdir('.'))):
    print(TRAIN_EXTRA_PATH, " existe")
    df = pd.read_csv(TRAIN_EXTRA_PATH)
else:
    counter = 0
    pbar = tqdm(total=70328)
    data = []
    other_labels = defaultdict(int)
    # Iterate over the XML file (since parsing it in one run might blow up the memory)
    for event, elem in etree.iterparse(PROTEINATLAS_XML_PATH, events=('start', 'end', 'start-ns', 'end-ns')):
        if event == 'start':
            if elem.tag == "data" and len({"location", "assayImage"} - set([c.tag for c in elem.getchildren()])) == 0:
                labels = []
                assay_image = None
                for c in elem.getchildren():
                    if c.tag == 'assayImage':
                        assay_image = c
                    if c.tag == 'location':
                        if c.text in name_to_label_dict:
                            label = name_to_label_dict[c.text]
                            if type(label) is int:
                               labels.append(label)
                            else:
                                for l in label:
                                    labels.append(l)
                        else:
                            other_labels[c.text] += 1
                if not labels:
                    # Let's ignore images that do not have labels
                    continue
                for image in assay_image.getchildren():
                    if len(image.getchildren()) < 4 or image.getchildren()[-1].text is None:
                        continue
                    image_url = image.getchildren()[-1].text
                    assert "blue_red_green" in image_url
                    for channel, color, object_ in zip(image.getchildren()[:-1], ["blue", "red", "green"], ["nucleus", "microtubules", "antibody"]):
                        assert channel.text == object_
                        assert channel.attrib["color"] == color

                    # "https://v18.proteinatlas.org/images/4109/24_H11_1_blue_red_green_yellow.jpg" -> "4109/24_H11_1"
                    data.append(["/".join(image_url.split("/")[-2:]).replace("_blue_red_green.jpg", ""), " ".join(str(x) for x in sorted(labels, reverse=True))])
                    counter += 1
                    pbar.update()
            # This is necessary to free up memory
            elem.clear()
    print(counter)
    # Samples are also labeled with 'nucleus', which can not be translated into official labels
    print(other_labels)


print(df.head())

hpa_base = "https://v18.proteinatlas.org/images"
titles = ["RGBY", "Segmentation", "Intensity Green", "Intensity Blue", "Intensity Red", "Intensity Yellow"]
urls = ["/{}_blue_red_green_yellow.jpg", "_cell_segmentation/{}_segmentation.png", "/{}_green_lut.jpg", "/{}_blue_lut.jpg", "/{}_red_lut.jpg", "/{}_yellow_lut.jpg"]
_, axes = plt.subplots(nrows=1, ncols=len(urls), figsize=(5 * len(urls), 5))
for index, (title, url) in enumerate(zip(titles, urls)):
    image = io.imread(hpa_base + url.format(df.loc[0, "Id"]))
    axes[index].imshow(image)
    axes[index].set_title(title)
    axes[index].figure
    plt.show()  # will display fig1 and fig2 in different windows
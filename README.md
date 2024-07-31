# MultiFusionNet
Official repo of the work "MultiFusionNet: Multilayer Multimodal Fusion of Deep Neural Networks for ChestX-Ray Image Classification"

# CoV-Pneum Dataset
We prepared a new dataset called Cov-Pneum by processing
and merging three well-known publicly available
sub-datasets from Kaggle (Chowdhury et al. 2020; Rahman
et al. 2021; Kaggle 2021a, b). These sub-datasets encompass
diverse sources of origin for generating CXR images.
For instance, the sub-database (Chowdhury et al. 2020) has six
sources of origin. Four of these are utilized for COVID-19
CXR images, while two are Kaggle open sources, providing
pneumonia and normal CXR images. These sources are as
follows:

Italian Society of Medical and Interventional Radiology
(SIRM)COVID-19 Database: The SIRMCOVID-19 database
(I S.I.S.O.M.A. 2020) includes 384 radiographic images
(CXR and CT) with the varying resolution, comprising 94 chest
X-ray images and 290 lung CT images. Here, the author took
CXR images only.

Novel Corona Virus 2019 Dataset: a public GitHub
database (Monteral 2020) comprising 319 radiographic
images of COVID-19, MERS, SARS, and ARDS sourced
from published articles and online resources. The database
includes 250 COVID-19-positive CXR images and 25
COVID-19-positive lung CT images with diverse resolutions.
However, this study focuses on 134 unique COVID-19 positive
CXR images, distinct from those in the authors’ created
database derived from various articles.

COVID-19 positive chest X-ray images from different
articles: The authors, motivated by the GitHub database,
explored over 1200 articles within two months. Notably,
the GitHub database lacked comprehensive X-ray and CT
images, containing only a limited number with random
sizes. To address this, the authors painstakingly gathered and
indexed images from recent articles and online sources, comparing
them with the GitHub database to prevent duplication.
They successfully acquired 60 COVID-19-positive chest Xray
images from 43 new articles and 32 positive chest X-rays
images from Radiopaedia, none of which were present in the
GitHub database (Chowdhury et al. 2020; Radiopedia 2020).
COVID-19 chest imaging at thread reader: A Spanish
physician contributed 103 images representing 50 cases to the
Chest imaging thread reader (Imaging C 2020) showcasing diverse resolutions. The normal and viral pneumonia subdatabases,
comprising 1579 and 1485 X-ray images were
crafted using data from the RSNA-Pneumonia-Detection-
Challenge and Kaggle’s Chest X-ray Images databases,
respectively.


RSNA Pneumonia Detection Challenge: Included in this
database is normal chest X-ray images without lung infections
and non-COVID pneumonia images.
Chest X-Ray Images (pneumonia): The widely-used Kaggle
chest X-ray database contains 5247 images capturing
normal, viral, and bacterial pneumonia, ranging in resolution
from 400 to 2000p (Mooney 2018). Among these, 3906
images depict subjects with pneumonia (2561 for bacterial
and 1345 for viral pneumonia), while 1341 images are of
normal subjects.

A similar approach is adopted to generate sub-datasets
from various sources, encompassing diverse medical image
modalities such as CXR and CT scans. Each of the three
sub-datasets exclusively contain CXR images showcasing
distinct resolutions and representing individuals from varied
regions and age groups. By merging these sub-datasets, the
Cov-Pneum dataset consists of a total of 21,272 CXR images
categorized into COVID-19 (lung infected with COVID-
19 virus), Pneumonia (viral Pneumonia, non-COVID-19
Pneumonia and COVID-19 Pneumonia infected lung), and
Normal (clear lung) classes. The dataset contains 4296
COVID-19 images, 5824 Pneumonia images, and 11,152
Normal images. To improve the quality of the CXR images,
we applied image scaling and pre-processing operations.

1. ChowdhuryME, Rahman T, Khandakar A,Mazhar R, KadirMA, Mahbub
ZB et al (2020) Can ai help in screening viral and covid-19
pneumonia? IEEE Access 8:132665–132676.
2. Kaggle (2021a) Chest x-ray images (pneumonia). https://v.ht/WwR25
3. Kaggle (2021b) Covid-net open source initiative—covidx cxr-3 dataset.
https://www.kaggle.com/andyczhao/COVIDx-cxr2?select=test

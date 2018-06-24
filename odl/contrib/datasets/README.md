# Datasets

Reference datasets with accompanying ODL geometries etc.

## Content:

* `mri`

  Magnetic Resonance Imaging (MRI).
  * `tugraz`

    MRI data as provided by TU Graz (License [CC BY 4.0](./licenses/CC_BY_40)). The data is so called multi-channel or multicoil data, which implies that the reconstruction is non-linear.
    * `mri_head_data_4_channel`
    * `mri_head_data_32_channel`
    * `mri_knee_data_8_channel`
* `ct`

  Computed Tomography (CT)
  * `fips`

    CT data as provided by FIPS (License [CC BY 4.0](./licenses/CC_BY_40)). The data is non-human and high resolution.
    * `walnut_data`
    * `lotus_root_data`

    CT data as provided by Mayo Clinic. The data is from a human and of high resolution (512x512). To access the data, see [the webpage](https://www.aapm.org/GrandChallenge/LowDoseCT/#registration). Note that downloading this dataset requires signing up and signing a terms of use form.
    * `load_projections`
    * `load_reconstruction`
* `images`

  Two dimensional images.
  * `cambridge`

    Various images as provided by the University of Cambridge, see http://store.maths.cam.ac.uk/DAMTP/me404/data_sets/ (License [CC BY 4.0](./licenses/CC_BY_40))
    * `brain_phantom`
    * `resolution_phantom`
    * `building`
    * `rings`
    * `blurring_kernel`

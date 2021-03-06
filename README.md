# NBB's Extension :  morphing  between images using cross-domain correspondence

This is our PyTorch implementation for the NBB's Extension :  morphing sequence between images using cross-domain correspondence.

It was made as a part of GAN workshop at Tel-Aviv University ,Under the guidance of Prof. Daniel Cohen-Or and Sharon Fogel.

This extention code to the NBB's algorithm was written by Tal Hadar , Matan Richker, Jonatan Hadas


**NBB's Extension :  morphing sequence between images using cross-domain correspondence: [Paper](https://github.com/talhadar1/NBBs_extension_morphing_sequence_using_cross_domain_correspondence/blob/master/NBB_s_Morphing___using_a_cross_domain_correspondence_to_morph__between_images.pdf)**


<img src="first_page_sample.png" width="800" />
<img src="ezgif.com-video-to-gif.gif" width="250" />



The original Neural Best-Buddies: Sparse Cross-Domain Correspondence code written by:
[Kfir Aberman](https://kfiraberman.github.io/), [Jing Liao](https://liaojing.github.io/html/), [Mingyi Shi](https://rubbly.cn/), [Dani Lischinski](http://danix3d.droppages.com/), [Baoquan Chen](http://www.cs.sdu.edu.cn/~baoquan/), [Daniel Cohen-Or](https://www.cs.tau.ac.il/~dcor/), SIGGRAPH 2018.

## Prerequisites
- Linux or macOS
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Run
- Run the distance calculation algorithm (demo example)
```bash
#!./script.sh
python3 main.py --sourceImg ./input/Pics_100_aligined/00.png --targetImg ./input/Pics_100_aligined/99.png  --input_dir input --name Pics_100_test --k_final 5 --k_per_level 10 --fast --src_index 72  --trg_index 15
```

- Run the distance calculation algorithm (demo example - After the disance calculated)
```bash
#!./script.sh
python3 main.py --sourceImg ./input/Pics_100_aligined/00.png --targetImg ./input/Pics_100_aligined/99.png  --input_dir input --name Pics_100_test --k_final 5 --k_per_level 10 --fast --data_stored --src_index 72  --trg_index 15
```
The option `--k_final` dictates the final number of returned points. The results will be saved at `../results/`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory.

### Output
-morph sequence:
- morph_path.txt					, morph_dijkstra.txt
- morph_path_domain_manipulation.txt, morph_dijkstra_domain_manipulation.txt 

### Next Step
- Take this path files and insert them as is to the morphing folder and execute the morphing program.

### Tips
- you can set the domain manipulation parameter to any disierd number. it will divide the distancess between intra-domain images.
- If you are running the algorithm on a bunch of pairs, we recommend to stop it at the second layer to reduce runtime (comes at the expense of accuracy), use the option `--fast`.
- If the images are very similar (e.g, two frames extracted from a video), many corresponding points might be found, resulting in long runtime. In this case we suggest to limit the number of corresponding points per level by setting `--k_per_level 20` (or any other desired number)



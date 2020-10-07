---
title: Feature descriptors
keywords: scale-invariant keypoint detection, Laplacian, Difference of Gaussians, DoG, Scale-Invariant Feature Transform, SIFT, Histogram of Oriented Gradients, HoG
order: 6 # Lecture number for 2020
---

- [Scale-Invariant Keypoint Detection](#scale-invariant-detection)
	- [Laplacian](#laplacian)
	- [Difference of Gaussians](#dog)
- [Scale-Invariant Feature Transform](#sift)
    - [Invariant Local Features](#invariant-local-features)
    - [SIFT](#sift-descriptor)
    - [SIFT Descriptor Formation](#sift-descriptor-formation)
    - [Analysis](#sift-analysis)
    - [Application](#sift-application)
- [Histogram of Oriented Gradients](#hog)
    - [HoG](#hot-intro)
    - [Normalization](#norm)
    - [Visualizing HoG](#visual)
    - [Differences between HoG and SIFT](#diff-hog-sift)

## 6.1 Scale-Invariant Keypoint Detection <a name="scale-invariant-detection"></a>
We have previously discussed various feature detection methods like the Harris corner detector. Yet, we run into a problem with using these methods on their own once we have a large difference in the scale between our images. For example, when using the Harris corner detector, we search for our features using a set window size, but we can see how a difference in scale could cause the same corner in a scaled up image to no longer fit our feature criteria.

<div class="fig figcenter fighighlight">
  <img src="https://i.imgur.com/AxYbpQ1.png">
  <div class="figcaption"> A change in scale causes the same corner to appear only as an edge</div><br>
</div>

Thus, our goal is to be able to find a method for “scale invariance detection” such that we can recognize the same features independently within these images, regardless of their difference in scale. To do so, we need to not only look at a region, but also take into consideration the various neighborhoods of differing sizes around this region. It is important to estimate the size of the neighborhood that can lead to best matching between images of different scale.

We can create a scale invariant function, meaning that it will return the same value for corresponding regions even at different scales. The function will respond to contrast (sharp local intensity change) in the image and we can use the local maximums to determine the region size relative to scale.

<div class="fig figcenter fighighlight">
  <img src="https://i.imgur.com/vGKzGNe.png">
  <div class="figcaption"> The function will need to be covariant with image scale.</div><br>
</div>
<div class="fig figcenter fighighlight">
  <img src="https://i.imgur.com/1d916WZ.png">
  <div class="figcaption"> The function should also ideally have one stable sharp peak.</div><br>
</div>

There are multiple possibilities for scale invariant detection functions, typically of the form:

$F = Kernel * Image$

We can take a more in depth look at two main types of these kernels: Laplacian and Difference of Gaussians.


### 6.1.1 Laplacian <a name="laplacian"></a>

The Laplacian is an example of a kernel for scale-invariant keypoint detection defined as follows:

$L = \sigma^2(G_{xx}(x,y,\sigma) + G_{yy}(x,y,\sigma))$

where G is the Gaussian function:

$G(x,y,\sigma) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{x^2+y^2}{2\sigma^2}}$

By maximizing the Laplacian function within an image we can find the maximum contrast from a region.

<div class="fig figcenter fighighlight">
  <img src="https://i.imgur.com/SI33SiB.png">
  <div class="figcaption"> We define the "characteristic scale" as the scale that produces the peak of Laplacian response.</div><br>
</div>

This will allow us to compare different sized regions around a key point to determine our scale in an image independently through local maximums. Therefore the Laplacian can be used alongside the Harris feature detector to allow for scale invariant feature detection.

#### Harris-Laplacian Scale Invariant Detector:
1. Run the Harris corner detector at a predefined scale to retrieve initial key points and their corresponding region size in space (image coordinates).
2. Redefine the scale of the region around the key points by finding the local maximum of the Laplacian.

<div class="fig figcenter fighighlight">
  <img src="https://i.imgur.com/kSKhFj0.png">
  <br>
  <div class="figcaption"> We can see the difference in feature detection the addition of the Laplacian makes.</div>
</div>


### 6.1.2 Difference of Gaussians <a name="dog"></a>
Difference of Gaussians (DoG) is another method for scale-invariant keypoint detection. In DoG, the input image is blurred with Gaussian kernels of increasing widths (i.e. standard deviations), and the difference is taken between successive Gaussian-blurred images.

For example, if we have an input image $I$ and Gaussian kernels $G(k_1\sigma)$, $G(k_2\sigma)$, and $G(k_3\sigma)$, where $k_1 < k_2 < k_3$, then performing DoG would result in the images $(G(k_2\sigma) - G(k_1\sigma)) * I$ and $(G(k_3\sigma) - G(k_2\sigma)) * I$.

<figure>
  <center><img src="https://i.imgur.com/6Bht1Ri.png"/></center>
  <figcaption>The initial step of DoG applies Gaussian kernels of increasing width to the input image then takes difference between consecutive blurred images.
  </figcaption>
</figure>

In DoG, keypoints are defined as any local extrema within space and scale, i.e. pixels that have smaller intensity or greater intensity than their 8 neighbors within the same DoG layer, the 9 pixels centered on the corresponding pixel in the previous DoG layer, and the 9 pixels centered on the corresponding pixel in the following DoG layer.

<figure>
  <center><img src="https://i.imgur.com/1foU3MJ.png"/></center>
  <figcaption>DoG selects pixels that are local maxima       within space and scale as keypoints. Such extrema have higher or lower intensity than all 26 of their neighbors.
  </figcaption>
</figure>

Applying DoG with just two different Gaussians results in the detection of fine details of the input image. As the number of Gaussians increases, we detect larger, more general details.

<figure>
  <center><img src="https://i.imgur.com/JNeRSmT.png"/></center>
  <figcaption>DoG is applied to a video with two, three, or four Gaussians (i.e. one, two, or three differences). The white pixels in the resulting images are considered keypoints. As you can see, the result of DoG for k1-k2 shows fine-grained features, whereas the result of DoG for k1-k4 shows much broader details.
  </figcaption>
</figure>

## 6.2 Scale-Invariant Feature Transform <a name="sift"></a>

Now that we know how to detect key regions/ points in an image for image matching, the next question that comes to mind is: How do we describe such points for image matching? To do so, we first have to understand what a **point descriptor** is. A point descriptor is a vector that summarizes the content of a particular keypoint neighborhood. We can then use this point descriptor for matching and comparing key points across images. A point descriptor should be:
1. Invariant (i.e. to illumination changes, scale, rotation, etc.)
2. Distinctive (different neighborhoods/ key points produce different descriptors)

<figure>
  <center><img src="https://imgur.com/L0pM6Le.png" alt="my alt text"/></center>
  <figcaption>Point descriptors are vectors that summarize the content of a keypoint neighborhood.</figcaption>
</figure>

### 6.2.1 Invariant Local Features <a name="invariant-local-features"></a>
Recall that utilizing invariant local features is an alternative to using global features/ templates. With this approach, image content is transformed into local feature coordinates that are invariant to translation, rotation, scale, and other imaging parameters.

<figure>
  <center><img src="https://imgur.com/np1wjkc.png" alt="my alt text"/></center>
  <figcaption>Regardless of translation, rotation, scale, and other parameters, the common local invariant features are still able to be extracted from the two (rather different) images of this toy truck.</figcaption>
</figure>

Consider how we would approach implementing rotation invariant descriptors. We have two main options, namely:

1. Upon detecting the keypoint neighborhood, normalize the patches by rotating them into a canonical orientation.
2. Construct a rotation invariant descriptor (i.e. make the computation rotation invariant).

For the purposes of this lecture, we will adopt Option 2. Suppose we are given a keypoint and its neighborhood scale/ size from an arbitrary detector (i.e. DoG). We will select a characteristic orientation for the keypoint by computing the most prominent gradient in the neighborhood. Now, we can describe all features ***relative*** to this dominant orientation. However, note that this inadvertently causes features to be rotation invariant, which is what we intended to show. 

To observe this, consider a particular patch/ neighborhood in image $A$ and image $B$. If calculating the features in $A$ with respect to $A$\'s characteristic orientation yields the same result as calculating the features in $B$ with respect to $B$\'s characteristic orientation, we will have found the same features. In other words, if the keypoint appears rotated in another image, the features will be the same, since they are ***relative*** to the characteristic orientation!

<figure>
  <center><img src="https://imgur.com/twoeqoB.png" alt="my alt text"/></center>
  <figcaption>Option #1 approach: Normalize patches of this cute kitty by rotating them -- we are still able to identify the same patches.</figcaption>
</figure>

<figure>
  <center><img src="https://imgur.com/tUWY1mH.png" alt="my alt text"/></center>
  <figcaption>Option #2 approach: Construct a rotation invariant descriptor using a keypoint & DoG scale and selecting the characteristic orientation. All features will now be described relative to this orientation.</figcaption>
</figure>

### 6.2.2 SIFT Descriptor <a name="sift-descriptor"></a>

Heeding this phenomenon with the rotation-invariant descriptors, we will now apply this concept when constructing the **SIFT descriptor**, where SIFT is an abbreviated term for "Scale-Invariant Feature Transform". A SIFT descriptor is a gradient-based descriptor that captures texture in a particular keypoint neighborhood. The process for constructing the SIFT descriptor is as follows:

1. Blur the image and extract the image patch from the keypoint neighborhood.
2. Take image gradients over the keypoint neighborhood.
3. Rotate the gradient directions ***and*** locations by $-\theta$ (in order to become rotation invariant). 

For step #3, this is intended to compensate for the patch rotation by subtracting back from the gradients, effectively cancelling out the rotation and having the gradients expressed at locations relative to the keypoint orientation $\theta$. *Note: We could have rotated the entire image by $-\theta$, but it is computationally more expensive.*

<figure>
  <center><img src="https://imgur.com/iz29BhY.png" alt="my alt text"/></center>
  <figcaption>SIFT descriptor: Depicted is an image patch of 8x8 pixels. We take the image gradients in this neighborhood and rotate the coordinates AND gradients by (-keypoint orientation).</figcaption>
</figure>

### 6.2.3 SIFT descriptor formation <a name="sift descriptor-formation"></a>
Can we simply stack all gradients into a single vector as the descriptor? Yes, but this would be a little problematic because using precise gradient locations is fragile. We would like to allow some "slop" in the exact pixel configurations within the image, thus a better option will be an orientation histogram. 
From each keypoint neighborhood, we can compute a histogram of gradient orientations. Every pixel votes for a certain angle that its gradient is pointing at. With the histogram, we capture information about the texture in the patch while allowing the precise location of the gradient to be looser. 

In the example shown below, we have eight entries in the histogram, which means we convert the gradients into a vector with 8 dimensions.

<figure>
  <center><img src="https://imgur.com/DCevyK5.png" alt="my alt text"/></center>
  <figcaption>Instead of stacking all gradients into a long vector, we use histogram to capture information from the patch without constraining the precise locations.</figcaption>
</figure>

Besides converting the whole patch into one single histogram and discarding all positional information, we can also create an array of histograms. In the following example, we use a 4x4 array of histograms. Each element in the array is transformed from a 2x2 cell in the patch. 

One pixel can contribute not only to its own cell but also to its neighboring cells. In this case, we can split the gradient of this pixel to different cells weighed by its distance to the center of these cells.

<figure>
  <center><img src="https://imgur.com/jC0tomK.png" alt="my alt text"/></center>
  <figcaption></figcaption>
</figure>

In practice, the author of SIFT did extensive experiments and founds out that the best result is obtained by using 8 orientation bins and a 4x4 histogram array.

As a recap, given a patch, we first compute its gradients. Then we normalize the gradients with respect to the predominant orientation, and we divide the patch into a 4x4 array. Within each cell of the array, we compute a histogram of gradient orientations. Each histogram has 8 bins and can be seen as an 
8-dimensional vector. Thus we end up with a descriptor that is a 128(=8x4x4)-dimensional vector. 

The descriptor is rotation-invariant because we normalized all the orientations to the predominant orientation; it is also scale-invariant because we use the best scale found by the key point detector(e.g., DoG detector). 

By measuring similarities of SIFT descriptors, we can compare patches from different images and therefore match key points. For example, the Euclidean distance between descriptor vectors gives a good measure of similarity.

We can play a few more tricks to make the descriptor robust to illumination changes. First, the descriptor is constructed of gradients, so it's already invariant to changes in brightness. (Adding a constant to all pixels doesn't change any gradient!) However, if we have a higher-contrast photo, the magnitude of all gradients will be increased linearly. Thus we can normalize the histogram (make the magnitude of the descriptor vector =1.) to make it contrast invariant. 
<figure>
  <center><img src="https://imgur.com/4zyq2i5.png" alt="my alt text"/></center>
  <figcaption>Example of glare. From robsonforensic.com.
  </figcaption>
</figure>

In addition, very large image gradients are usually from unreliable 3D illumination effects. To reduce their effect, we can clamp all values in the vector to be $\leq 0.2$. Again, this is an experimentally tuned value. Finally, normalize the vector again, and we end up with a descriptor vector which is fairly invariant to illumination conditions.

### 6.2.4 Analysis of SIFT descriptor<a name="sift-analysis"></a>
With ideas about how to compute SIFT descriptor, we can move on to some analysis of the descriptor. Some of the parameters of the descriptor calculation were tuned experimentally by the authors of the paper. One of them is how many orientations do we need in our histogram.

<figure>
  <center><img src="https://imgur.com/WPrRCC3.png" alt="my alt text"/></center>
  <figcaption>David G. Lowe, "Distinctive image features from scale-invariant keypoints," International Journal of Computer Vision, 60, 2 (2004), pp. 91-110
  </figcaption>
</figure>

The plot above shows in the vertical axis the percentage of correctly matched points and the horizontal axis shows the size of the array. For example, a 4 here means we divide the patch into a four by four array. As you can see, the best results are obtained when we use a four by four array and eight histogram orientations.

In these experiments, the authors try to match features from images after randomly changing their scale and orientation and by adding different level of noise. The idea was to see whether the matching was stable when adding different levels of noise.

<figure>
  <center><img src="https://imgur.com/unUJEMo.png" alt="my alt text"/></center>
  <figcaption>David G. Lowe, "Distinctive image features from scale-invariant keypoints," International Journal of Computer Vision, 60, 2 (2004), pp. 91-110
  </figcaption>
</figure>

The horizontal axis shows the level of noise added to the image and the vertical axis shows the percentage of correctly matched key points. We can see that the performance is quite stable when noise level increases.

The authors also try to check whether the matching was stable across different viewpoints.

<figure>
  <center><img src="https://imgur.com/G0zC7Ce.png" alt="my alt text"/></center>
  <figcaption>David G. Lowe, "Distinctive image features from scale-invariant keypoints," International Journal of Computer Vision, 60, 2 (2004), pp. 91-110
  </figcaption>
</figure>

The horizontal axis shows the viewpoint angle change from one image to the other, from zero degrees to fifty degrees. Again, the vertical axis shows the correctly matched key points. The performance here is still somewhat stable for smaller viewpoint angle changes and it degrades gracefully for larger viewpoint angle changes.

Another experiment tries to evaluate the distinctiveness of the feature descriptor. The idea here was to evaluate through image search. For instance, someone wants to compare one image to a large database of images and try to find the closest one. If the database of images is very large, then the features need to be very precise in order to be able to distinguish between one image and the other.

<figure>
  <center><img src="https://imgur.com/IFdVWi6.png" alt="my alt text"/></center>
  <figcaption>David G. Lowe, "Distinctive image features from scale-invariant keypoints," International Journal of Computer Vision, 60, 2 (2004), pp. 91-110
  </figcaption>
</figure>

The authors tried to vary the size of the database and the results here seem to be quite stable across different database sizes.

### 6.2.5 Application of SIFT descriptor<a name="sift-application"></a>

When it comes to applications, the authors showed several interesting examples. 

<figure>
  <center><img src="https://imgur.com/4vYx2jC.png" alt="my alt text"/></center>
  <figcaption>David G. Lowe, "Distinctive image features from scale-invariant keypoints," International Journal of Computer Vision, 60, 2 (2004), pp. 91-110
  </figcaption>
</figure>

Here the idea is to localize the train and the frog in the pictures on the left, within the image in the middle. By using those ideas of local feature matching, they were able to localize the frog and two instances of the train. What's interesting here is that the localization was possible even in the presence of heavy occlusion. That's because the matching is done with local parts of the object, rather than with the global template.

<figure>
  <center><img src="https://imgur.com/fFJxGIY.png" alt="my alt text"/></center>
  <figcaption>David G. Lowe, "Distinctive image features from scale-invariant keypoints," International Journal of Computer Vision, 60, 2 (2004), pp. 91-110
  </figcaption>
</figure>

Here's another example of a complex object search in a large scene. The idea was to localize these four objects in this picture. The templates given to the algorithm were took from different viewpoints than the test image, but still it's able to localize the objects. 

This idea of localizing objects using templates was further extended later on.

<figure>
  <center><img src="https://imgur.com/VLRddIc.png" alt="my alt text"/></center>
  <figcaption>
  </figcaption>
</figure>

For example, localizing objects in videos by Sivic and Zisserman, or localizing objects with 3D formation, as well as localizing objects not only with bounding boxes, but with their contours.

<figure>
  <center><img src="https://imgur.com/xNvC0wS.png" alt="my alt text"/></center>
  <figcaption>
  </figcaption>
</figure>

There's also applications to panorama stitching where given a collection of images like the one in the top. We are able to assemble them into a single picture, like the one in the middle. With some more processing, we can hide those boundaries between different images and obtain the image at the bottom.

<figure>
  <center><img src="https://imgur.com/B2n8z9j.png" alt="my alt text"/></center>
  <figcaption>
  </figcaption>
</figure>

Another application for SIFT and feature matching is stereo vision. In particular, when the two cameras are not so close to each other, but rather far apart from each other, this is called wide baseline stereo. This kind of stereo setup requires more robust feature matching, like the one SIFT can provide.

In addition, there are also applications in:

+ Mobile Robot Navigation
+ Motion Tracking
+ 3D Reconstruction
+ ...

In summary, the SIFT descriptor has wide applications and provides quite robust feature matching results.

## 6.3: Histogram of Oriented Gradients <a name="hog"></a>

We recall that a feature descriptor is an algorithm that extracts the most useful information from an image and disregards the extraneous details. It does so by taking in an image as input, and outputting a robust feature set that allows object forms within the image to be discriminated.  

However, some of the major challenges that robust feature descriptors must circumvent include: 

+ Accomodating a wide range of object pose and large variations in object appearance
+ Handling cluttered backgrounds and varying levels of illumination
+ Ensuring fast computation for implementation in low-power devices, such as mobile

### 6.3.1 Introduction to HoG<a name="hog-intro"></a>

One interesting method for describing objects and features is Histogram of Oriented Gradients (HoG). Proposed at the Conference on Computer Vision and Pattern Recognition (CVPR) by Dalal & Triggs in 2005, the main idea was that local object appearance and shape can often be characterized rather well by the distribution of local intensity gradients or edge directions. 

<figure>
  <center><img src="https://i.imgur.com/IXKi87a.png" alt="my alt text"/></center>
  <figcaption>The left image displays an example image of a pedestrian included in the dataset. The right image displays the average gradient image found with the implementation of HoG.</figcaption>
</figure>


In the example pictured above,  Dalal & Triggs (2005) find the average gradient image over a database of 1,800 images of pedestrians with a large range of poses and variations of backgrounds, similar to the leftmost image. The result was an outline of a person displayed on the rightmost image, hinting that the edges and gradients in an image can be strong hints for object detection. Evidently, HoG detectors were most sensitive to silhouette contours in the experiment (especially head, shoulders, and feet).    

HoG takes a similar approach to SIFT. However, it is different in the sense that the descriptor is calculated over a larger region in the image. For example, suppose we have an image such that the window contains a person. To implememt HoG, we would take the following series of steps:

1. Divide the image window into small spatial regions, or small cells. This is equivalent to the arrays in SIFT. 
2. Unlike SIFT, the cells do not have to be rectangular; they can be either rectangular or radial
3. Just like in SIFT, we can calculate a histogram of gradient orientations over the pixels inside the cell. Each cell accumulates a weighted local 1-D histogram of gradient directions over the pixels of the cell. 

<!-- ![](https://i.imgur.com/iAJfb7V.png =100x) -->

<figure>
  <center><img src="https://i.imgur.com/wS0wWhw.png" alt="my alt text"/></center>
  <figcaption>A histogram of gradient orientations can be calculated with the patch of image shown above. </figcaption>
</figure>

However, the key difference between HoG and SIFT is that we can construct arrays over larger windows in an image with HoG. In the example shown below, an array is constructed over the region of an image large enough to contain a person. This means that while SIFT focuses on single key points, HoG can focus on larger windows that may contain an object.



<figure>
  <center><img src="https://i.imgur.com/hdjamtq.png" alt="my alt text"/></center>
  <figcaption>In this example, we visualize the orientations of the gradients within each cell. </figcaption>
</figure>

### 6.3.2 Normalization <a name="norm"></a>
In order to achieve better invariance to illumination and shadowing, it is necessary to perform normalization. We can recall that in SIFT, invariance to illumination is reached by normalizing the histograms independently in each cell. HoG adds another dimension by additionally normalizing over blocks. **Blocks** are larger spatial regions that contain multiple cells. The key idea is that HoG normalizes not only at the level of cells but additionally at the level of blocks. Another way of visualizing this is that the "energy" of local histograms is accumulated over the blocks, in effect normalizing all of the cells in the block. When neighboring cells exhibit different contrast, normalizing over blocks of multiple cells increases robustness to illumination changes.

<figure>
  <center><img src="https://i.imgur.com/Hpe5zVX.png" alt="my alt text"/></center>
  <figcaption>Rectangular (R-HOG) and Circular (C-HOG) blocks made up of multiple neighboring cells </figcaption>
</figure>

In their research on HoG, Dalal and Triggs (2005) found that having the blocks overlap increases performance significantly. Using overlapping blocks was found to decrease the miss rate by approximately 5%. 

<figure>
  <center><img src="https://i.imgur.com/F6xYar9.png" alt="my alt text"/></center>
  <figcaption>Overlapping blocks, which are depicted in red, result in higher performance for HoG. The block overlap shown here is half the block size.  </figcaption>
</figure>


Their research also discovered that 3x3 cell blocks of 6x6 pixel cells performed best, at least in the case of detecting people in images.

<figure>
  <center><img src="https://i.imgur.com/yVannYs.png" alt="my alt text"/></center>
  <figcaption> Graph depicting the miss rate in percent, indicated by the height of the bars, for varying block and cell sizes. 
  
  Dalal, Navneet, and Bill Triggs. “Histograms of oriented gradients for human detection.” Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on. Vol. 1. IEEE, 2005. </figcaption>
</figure>

One way to perform contrast normalization at the block level is to combine the values in the histogram bins for all of the cells in the block into a single array. The L2 or Euclidian norm can then be calculated by taking the square root of the sum of the squares of all of the elements in the array, as follows:
$$||v|| = \sqrt{v_1^2 + v_2^2 + ... + v_n^2} = \sqrt{v^Tv}.$$

The normalized array is then derived by dividing the array by the norm calculated above:
$$\frac{v}{||v||} = \left[{\frac{v_1}{||v||}, \frac{v_2}{||v||}, ..., \frac{v_n}{||v||}} \right].$$

### 6.3.3 Visualizing HoG <a name="visual"></a>
It is helpful to visualize the HoG descriptor. In the following figure, image a) displays the average gradient image over all positive examples of images of people from the database of pedestrians described earlier. Image b) displays which cells contain positive evidence for the presence of a person. As can be seen in the image, the brighter cells correspond to the regions of the image that contain the outline of the head and shoulders. In contrast, image c) displays which cells contain evidence for a non-person or background. A test image containing a person is displayed as image d).

<figure>
  <center><img src="https://i.imgur.com/9epfOz3.png" alt="my alt text"/></center>
  <figcaption>Weighting the HoG descriptor in image e) with images b) and c) produces images f) and g), which highlight areas indicating evidence of a person or non-person, respectively.
  
  Dalal, Navneet, and Bill Triggs. “Histograms of oriented gradients for human detection.” Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on. Vol. 1. IEEE, 2005.
  </figcaption>
</figure>

The HoG descriptor of the test image d) is displayed in image e). The visualization of the descriptor displays a histogram for each cell in the window. The histogram in each cell is visualized by its strongest gradient orientations. 

Images f) and g) display the results of weighting the HoG descriptor in image e) by images b) and c), respectively. To obtain image f), image b), which shows the areas of the image that contain more evidence for a person, is multiplied by the HoG descriptor in image e). The orientations in f) show the outline of a person, as it highlights the areas and gradients that show more evidence for a person. Similarly, image g) is obtained by multipying image c), which displays the areas that contain evidence for a non-person, with the HoG descriptor in image e). Image g) thus highlights the portions that indicate evidence for a background or non-person.

This next example shows a visualization of the HoG descriptor for an image of a bicycle. The shape of bicycle frame, wheels, and seat can be seen in the visualization of the descriptor. Each histogram in the visualization corresponds to one cell in the window. Additionally, the visualization displays the top voted gradient orientations for each histogram.

<figure>
  <center><img src="https://i.imgur.com/LYQH7Pd.png" alt="my alt text"/></center>
  <figcaption>Image of a bicycle and the visualization of its corresponding HoG descriptor </figcaption>
</figure>

### 6.3.4 Differences between HoG and SIFT <a name="diff-hog-sift"></a>

There are many similarities between HoG and SIFT. However, there are some differences. The key difference between HoG and SIFT is that we can construct arrays over larger windows in an image with HoG. As a result, HoG is usually used to describe larger image regions while SIFT is used for key point matching and key point description. An important application of HoG is **object detection**, hence the need to use larger image regions. 
    
SIFT histograms are made rotationaly invariant since they are normalized with respect to the dominant gradient. HoG, however, is not and therefore it is not rotationally invariant. 

HoG needs normalization for better invariance to illumination and shadowing. Recall that SIFT achieves illumination invariance by normalizing the histograms independently inside each cell.
    
Unlike SIFT, the cells do not have to be rectangular; they can be either rectangular or radial. HoG introduces the idea of blocks. Blocks here are larger regions that contain multiple cells.Another difference is that HoG gradients are normalized not only within each cell but also within each block. 


<figure>
  <center><img src="https://i.imgur.com/zfp8b2l.png" alt="my alt text"/></center>
  <figcaption>An overview of Dalal & Triggs (2005) object detection chain.
  
  Dalal, Navneet, and Bill Triggs. “Histograms of oriented gradients for human detection.” Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on. Vol. 1. IEEE, 2005.
  </figcaption>
</figure>







The main conclusions of Dalal & Triggs (2005) were for good HOG performance, one should use fine scale derivatives, many orientation bins, and moderately sized, strongly normalized, overlapping descriptor blocks. Below image summarizes the effects of the various HOG parameters on overall detection performance. 


<figure>
  <center><img src="https://i.imgur.com/H6Tyi6X.png" alt="my alt text"/></center>
  <figcaption>Summary of the various HoG parameters from Dalal & Triggs (2005)[1]: (a) Using fine derivative scales increases performance. (b) Increasing the number of orientation bins up to about 9 bins increase performance. (c) The effect of different normalization schemes. 
</figcaption>
</figure>

### References


[1] Dalal, Navneet, and Bill Triggs. "Histograms of oriented gradients for human detection." Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on. Vol. 1. IEEE, 2005.

[2] David G. Lowe, "Distinctive image features from scale-invariant keypoints," International Journal of Computer Vision, 60, 2 (2004), pp. 91-110
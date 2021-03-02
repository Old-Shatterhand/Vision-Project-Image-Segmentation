# Vision-Project-Image-Segmentation

## Introduction

This is the final project for the course Neural Networks: Theory and Implementation (NNTI).
This project will introduce you to the Object Segmentation task.

In this project, we want you to implement different neural methods used in Object segmentation. You are expected to make use of concepts that you have learned in the lecture. The
project is divided into three tasks, the details of which you can find in the next pages.

### Repository

We have created a Github repository for this project. You will need to\
 • fork the repository into your own Github account.\
 • update your forked repository with your code and solutions.\
 • submit the report and the link to your public repository for the available code.

### Distribution of Points
The points in this project are equally distributed among the three tasks. You will able to score
a maximum of 10 points per task, resulting in a total of 30 points in the entire project. How 10
points are allotted for each task, is mentioned inside the description of sub-tasks inside tasks.
For example, task 1 has multiple sub-tasks in the iPython notebook with individual points.

## Task 1: First look on Object Segmentation (10 points)
While implementing Deep Learning models sometimes we make use of old codes and we try
to make slight changes to use them for our particular task. In this part, you are given a halfready code related to the object segmentation task. In the given iPython notebook there is
a simple object segmentation pipeline using the PASCAL VOC dataset. Fully understanding
and completing this task will give a good idea of how to proceed for the next tasks.

### To Do
• Follow the instructions in the notebook, complete the code, and run it.\
• Each part has a particular amount of points.\
• Update your repository with the completed notebook.

## Task 2: Recurrent Residual Convolutional Neural Network (10 points)
In this task, you will be implementing the ”Recurrent Residual Convolutional Neural Networkbased on U-Net (R2U-Net) for Medical Image Segmentation” paper and test it on cityscapes
dataset. This paper discusses an approach for image segmentation with three key benefits.
First, a residual unit helps when training deep architecture. Second, feature accumulation with
recurrent residual convolutional layers ensures better feature representation for segmentation
tasks. Third, it allows us to design better U-Net architecture with the same number of network
parameters with better performance for medical image segmentation.

### To Do
 • Read the paper in detail and understand the architecture in depth.\
 • Implement the exact model presented in the paper. You have the freedom to play with
the dimensions and other hyper-parameters but the model should look exactly in terms
of the number of filters, layers, etc.\
 • Your implementation should provide a dedicated script for each part of the pipeline,i.e.
separate script for data loading, the model, training, testing, etc. We require you to have
a well-documented code written in a modular fashion. Do not copy-paste fancy codes
from the internet without knowing what they are doing. The way how you write the code
will affect the grade for this part.\
 • To use the cityscapes dataset you can either use the inbuilt data loader from Pytorch(part
of the torchvision package) or you can write your own data loader.\
 • In the paper there is a specific section which deals with the metrics involved namely the
”Quantitative Analysis Approaches” section. You are expected to show the results for
each of the metrics mentioned there (roughly 5 different metrics) in your experiments.
As a general guideline please try to have at first a functioning pipeline then try to tune
the parameters. We are interested more in the way how you approached the problem so try to
elaborate on the details.

## Task 3: Challenge Task (10 points)
In this part also you will use the cityscapes dataset. In this third and final task of this project,
you are expected to :
• read multiple resources about the state-of-the-art work related to object segmentation.\
• try to come up with methodologies that would possibly improve your existing results\
• improve your results from task 2\
• You are expected to do follow all the instructions that were mentioned in Task 2 in terms
of how you will write your report and document your code.

Note: The task here should be a change in the model architecture, data representation, different
approach, or some other similar considerable change in your process pipeline. Please note that
although you should consider fine-tuning the model hyper-parameters manually, just doing that
does not count as a change here.

## General Guidelines
We have listed some guidelines which you will need to follow:
 • Write a well documented academic report. The report needs to be 4-8 pages (without
references with a NIPS format. You can have a look at Latex versions or in other formats.
We expect from you a .pdf file from you. The paper from task 2 could be a good example
of what we are expecting from you. The way how you divide it is up to you but we
roughly expect to have introduction, methodology, results, and conclusion sections. Of
course, you will have to cite every source that you use.\
 • The main focus of our grading will be your observations and analysis of the results. Even
though you might obtain bad results make comments on what could have gone wrong.\
 • We will be pretty strict in terms of plagiarism. We expect you to write your own code.
Of course, you can use external sources or documentation but there should be a clear
sign that you have done major edits to a ready code. Most importantly we will check for
plagiarism within groups. If we see any clear indication of plagiarism among groups both
of the groups will take a 0 for the whole project. Discussion with groups is allowed but
in terms of concepts but not directly with code.\
 • Try to separate the work among team members. For instance, Task 2 and Task 3 can be
done independently.\
 • If you have questions ask the tutors, ask in the Teams. If you are unsure
about something do not make assumptions on your own. You will have to
keep track of the posts in Team in case there is any update or change in the
project description.

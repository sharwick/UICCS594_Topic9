################################################################################
# Author: 		Shannon Harwick (sole author)
# Date:			Spring 2015
# Class:		UIC CS594 - Visualization
# Assignment:		Topic 9 - Blue Waters
################################################################################



##################
# Files
##################

- topic9_task1.pbs = Batch code to submit Task 1 Python code
- topic9_task1.py = Task 1 Python code
- task1_results.txt = Results for task 1 code (including notifications for
  intermediate steps)
- task1.err = Error log for task 1 code
- task1.out = Output log for task 1 code

- topic9_task2.pbs = Batch code to submit Task 2 Python code
- topic9_task2.py = Task 2 Python code
- task2_results.txt = Results for task 2 code (including notifications for
  intermediate steps)
- task2.err = Error log for task 2 code
- task2.out = Output log for task 2 code
- image.ppm = Composite 2D image resulting from compositing the images from
  produced by individual processes.

##################
# Notes
##################

- For part 1, the output indicates the overall mean is 372.072, which I 
  confirmed on my laptop by reading in data and computing mean of full
  data set directly.
- The input data has a 3 integer header, which I removed to construct my
  dataset of floats. 
- For task 2, I set alpha = .05 for all points and use Back-to-Front Compositing.
- Task 2 outputs image.ppm using the ppm format recommended in the assignment.


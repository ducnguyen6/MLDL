---
layout: post
title: Logistic Regression
tags:
- Regression
- Logistic
categories: Basic Machine Learning
description: Trong phần này mình sẽ trình bày về Logistic Regression và giải vài bài toán phân loại cơ bản.
---
## Giới thiệu về bài toán
Ta sử dụng bài toán [ex4](http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex4/ex4.html) trong khóa học Machine Learning của Andrew Ng. 
Bài toán đưọc mô tả như sau:
Cho tập dữ liệu [ ex4Data.zip ](http://openclassroom.stanford.edu/MainFolder/courses/MachineLearning/exercises/ex4materials/ex4Data.zip) 
chứa dữ liệu của 40 sinh viên đậu và 40 sinh viên rớt đại học. Mỗi mẫu \\( (x^{(i)}, y^{(i)}) \\)  chứa điểm số của 2 bài kiểm tra và kết quả thi của một sinh 1 viên.
Nhiềm vụ của ta là xây dựng một mô hình phân loại để ước lượng cơ hội đậu hay rớt của một sinh viên thông qua điểm của 2 bài kiểm tra.
Trong tập dữ liệu huấn luyện, ta có:
**a.** Cột đầu tiên của dữ liệu X đại diện cho điểm bài thi thứ 1 và cột thứ 2 đại diện cho điểm bài thi thứ 2.
**b.** Vector Y sử dụng '1' là lable cho sinh viên đậu và '0' là lable cho sinh viên rớt.
## Tìm lời Giải bài toán
Tập dữ liệu được biểu diễn như sau

![Data](/MLDL/assets/img/LRData.png)

---
layout: post
title: Logistic Regression
tags:
- Regression
- Logistic
categories: BasicMachineLearning
description: Trong phần này mình sẽ trình bày về Logistic Regression và giải vài bài toán phân loại cơ bản.
---
## Giới thiệu về bài toán
Ta sử dụng bài toán    [ex4](http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex4/ex4.html) trong khóa học Machine Learning của Andrew Ng.  

Bài toán đưọc mô tả như sau:   
Cho tập dữ liệu [ ex4Data.zip ](http://openclassroom.stanford.edu/MainFolder/courses/MachineLearning/exercises/ex4materials/ex4Data.zip)
chứa dữ liệu của 40 sinh viên đậu và 40 sinh viên rớt đại học. Mỗi mẫu \\((x^{(i)}, y^{(i)})\\)  chứa điểm số của 2 bài kiểm tra và kết quả thi của một sinh 1 viên.  
Nhiệm vụ của ta là xây dựng một mô hình phân loại để ước lượng cơ hội đậu hay rớt của một sinh viên thông qua điểm của 2 bài kiểm tra.   


Trong tập dữ liệu huấn luyện, ta có:   


**a.** Cột đầu tiên của dữ liệu X đại diện cho điểm bài thi thứ 1 và cột thứ 2 đại diện cho điểm bài thi thứ 2.   
**b.** Vector Y sử dụng '1' là lable cho sinh viên đậu và '0' là lable cho sinh viên rớt.   
## Biểu diễn dữ liệu

Tập dữ liệu được biểu diễn như sau   

![Data](/MLDL/assets/img/LRData.png)

Với điểm màu đỏ biểu diễn cho sinh viên là đậu và màu xanh là rớt.   

## Tìm lời giải.

### Hàm sigmoid
\\[ \sigma_{(t)} = \frac{1}{1+e^{-t}} \\]

Hàm này có đồ thị như sau:  

![Sigmoid](/MLDL/assets/img/LRSigmoid.gif)

- Có miền xác định \\( \mathbb{R} \\) và giá trị từ 0 đến 1.  
- Tồn đại đạo hàm tại mọi điểm.  
- Phù hợp cho việc phân loại (có là 1, không là 0)  

### Tìm hàm mất mát

Ta gọi   
\\( D = {(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(n)},y^{(n)})}, \forall x^{(i)} \in \mathbb{R}^d , y^{(i)} \in \{0,1\} \\)   
là tập dữ liệu đề cho.  
Mục tiêu của ta là cho dữ liệu của một sinh viên bất kỳ, dự đoán sinh viên đó đậu hay rớt.   
\\[  x^{(i)} \Rightarrow  \hat{h}^{(i)} \\]
Đặt \\( Y^{(i)} \\) là giá trị của \\( y^{(i)} \\) với đầu vào là \\( x^{(i)} \\)
\\[ Y^{(i)} \sim Bernouli(p,n) \\]
 Với:
\\[  p = P_{(y=1|x,w)} = \sigma_{(w^Tx)}  \\]
với \\( w = [w_0, w_1, ..., w_n]^T \\) là tham số cần ước lượng.
và \\( x = [1, x_1, ..., x_n] \\)
Để thuận tiện trong việc viết, ta đặt \\( \alpha^{(i)} = \sigma (w^Tx^{(i)}) \\)
 \\[ q = P_{(y=0|x^{(i)},w)} = 1 - p = 1 - \alpha^{(i)} \\]
Từ (1) và (2) ta suy ra:
\\[ P_{(y^{(i)}|x^{(i)},w)} = (\alpha^{(i)})^{y^{(i)}}(1-\alpha^{(i)})^{1-y^{(i)}} \\]
Xét trên toàn bộ tập dữ liệu D
\\[ P(Y|W) = \prod_{i=1}^{n}(\alpha^{(i)})^{y^{(i)}}(1-\alpha^{(i)})^{1-y^{(i)}}   \\]
Tìm mô hình phù hợp để  \\(P\\) lớn nhất.
Áp dụng negative Maximize log likelihood.  \\[ L = -log(P_{(Y|w)})  \\]
Vì \\( P_{(Y|w)} \in (0,1) \Rightarrow -log(P_{(Y|w)}) > 0 \\)  
Lúc này ta được \\( L \\) làm một hàm lồi (convex function) nên ta có thể  áp dụng các bài phương pháp tối    ưu lồi (convex optimization) để giải quyết bài toán này.   

## Gradient Descent method

Gradient Descent là một phương pháp tối ưu sử dụng phổ  biến trong bài toán tối ưu lồi.
Xét khảo sát một hàm số như hình vẽ   

![LRGD](/MLDL/assets/img/LRGD.png)   
gọi \\( x\* \\) là điểm cực trị cần tìm của hàm \\(f_{(x)}\\)  
Nếu đạo hàm của hàm số tại \\(x_t: f'_ {x_t} > 0 \\)
thì \\(x_t\\) nằm về phía phải so với \\(x\*\\).   
Vậy muốn đến được \\(x\*\\) ta cần di chuyển \\(x_t\\) về phía trái.

Và ngược lại, nếu đạo hàm của hàm số tại \\(x_t: f'_ {x_t} < 0 \\) thì \\(x_t\\) nằm về phía trái so với \\(x\*\\).   
Vậy muốn đến được \\(x\*\\) ta cần di chuyển \\(x_t\\) về phía phải.   

Một cách tổng quát, ta cần cộng cho \\(x_t\\) một lượng \\( \Delta \\)ngược dấu với đạo hàm:
\\[ x_{t+1} = x_t + \Delta \\]   

Nếu \\(x_t\\) càng xa \\(x\*\\) thì f'_ {x_t} càng lớn nến lượng \\( \Delta \\) sẽ tỉ lệ với đạo hàm.   
Từ đó ta suy ra được: \\[ x_{t+1} = x_t - \alpha f'_ {x_t} \\]
Với \\( \alpha > 0 \\) gọi là learning rate.

Tổng quát với hàm nhiều biến ta có:
Với hàm \\( h_{(X)} = w_0 + x_1w_1 + ... + x_nw_n \\):
\\[ X_{t+1} = X_t -\alpha \nabla_X f_(X_t)  \\]

với \\( \nabla_X f_(X_t) \\) là gradient của \\(f\\) theo biến \\(X\\)

Ví dụ:
cho hàm số \\[f_{(x)} = \\]

## Newton's method

Để tìm cực trị của hàm \\( g_{(x)} \\), ta cần tìm nghiệm của phương trình \\( g'_ {x} = 0\\).   

Xuất phát từ định lý taylor:   
\\[ f_{(x)} = f_{(x_0)} + f'_ {x_0}.(x-x_0) \\]   
Tìm x để \\[ f_{(x)} = 0 \\] \\[ \Leftrightarrow  f_{(x_0)} + f'_ {x_0}.(x-x_0) = 0 \\]
\\[ \Rightarrow  x = x_0 - \frac{f_{(x_0)}}{f'_ {(x_0)}} \\]
Đặt \\( f = g' \\) thì nghiệm của phương trình \\( g'_ {x} = 0\\) là:
\\[ x_{t+1} = x_t - \frac{g' _ {(x_0)}}{g''_ {(x_0)}}\\]
Tổng quá hóa cho hàm nhiều biến:
\\[ X_{t+1} = X_t - \mathbb{H}^{-1} _ {x} \nabla _ {x} f_{(x_0)}\\]

Ví dụ:

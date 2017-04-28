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

### Tìm hàm mất mát ( *loss function* )

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
### negative log likelihood.
Với hàm số trên, việc tối ưu là rất khó vì khi số \\(n\\) lớn thì
giá trị của \\(P_{(y^{(i)}|x^{(i)},w)}\\) sẽ rât nhỏ.
Ta sẽ lầy logarit cơ số e của \\(P_{(y^{(i)}|x^{(i)},w)}\\) ( thường được gọi là *hàm likelihood* ). Sau đó lấy ngược dấu để  được một hàm số mới có giá trị lớn hơn và là một hàm lồi (convex function). Lúc này bài toán ta trở thành tìm giá trị nhỏ nhất của hàm mất mát (hàm này thường được gọi là *negative log likelihood* ).
\\[ J_{(w)} = -log(P_{(Y|w)}) = -\sum_{i=1}^{n}(y^{(i)} log(\alpha^{(i)}) + (1-\alpha^{(i)})log(1-\alpha^{(i)})) \\]
Vì \\( P_{(Y|w)} \in (0,1) \Rightarrow -log(P_{(Y|w)}) > 0 \\)  
Lúc này ta được \\( J_{(w)} \\) làm một hàm lồi nên ta có thể  áp dụng các bài phương pháp tối    ưu lồi (*convex optimization* ) để giải quyết bài toán này.   

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

**Ví dụ:**   
Cho hàm số \\[f_{(x)} = x^2 \\]
Với điểm ban đầu \\(x_0 = 2\\) và \\(\alpha = 0.6 \\) ta được:
\\[ f' _ {x} = 2x \\]
\\[ x_{1} = x_0 - \alpha f'_ {x_0} = x_0 - 0.6\times 2x_0 = -0.2x_0\\]
\\[ \Rightarrow x_t = (-0.2)^t \times x_0 \\]
Với t càng lớn thì giá trị \\(x_t\\) càng gần 0 nên kết quả của ta càng chính xác.  
Tuy nhiên nếu ta chọn \\(x_0 = 2\\) và \\(\alpha = 0.5 \\) thì:
\\[ x_{1} = x_0 - \alpha f'_ {x_0} = x_0 - 0.5\times 2x_0 = 0\\]
Vậy là ta đã tìm được giá trị cực trị của \\(f_{(x)}\\) ngay tại lần lặp đầu tiên.
Từ đó ta thấy việc chọn \\(\alpha\\) và \\(x_0\\) khác nhau sẽ ảnh hưởng đến kết quả và số lần lặp của thuật toán.

## Newton's method

Để tìm cực trị của hàm \\( g_{(x)} \\), ta cần tìm nghiệm của phương trình \\( g'_ {x} = 0\\).   

Xuất phát từ định lý taylor:   
\\[ f_{(x)} = f_{(x_0)} + f'_ {x_0}.(x-x_0) \\]   
Tìm x để \\[ f_{(x)} = 0 \\] \\[ \Leftrightarrow  f_{(x_0)} + f'_ {x_0}.(x-x_0) = 0 \\]
\\[ \Rightarrow  x = x_0 - \frac{f_{(x_0)}}{f'_ {(x_0)}} \\]
Đặt \\( f = g' \\) thì nghiệm của phương trình \\( g'_ {x} = 0\\) là:
\\[ x_{t+1} = x_t - \frac{g' _ {(x_0)}}{g'' _ {(x_0)}}\\]
Tổng quá hóa cho hàm nhiều biến:
\\[ X_{t+1} = X_t - \mathbb{H}^{-1} \nabla _ {x} f_{(X_t)}\\]
Với \\( H \\) là ma trận Hessian.

**Ví dụ:**  
Cho hàm số \\[ f_{(x)} = x^2 -2x + 1 \\]
với điểm ban đầu là \\(x_0 = 3\\) ta được:
\\[ f'_ {(x)} = 2x - 2 \\]
\\[ f'' _ {(x)} = 2 \\]
\\[ x_1 = x_0 - \frac{f'}{f''} = x_0 - \frac{2x_0-2}{2} = 1 \\]
Vậy với hàm bật 2 một biến thì chỉ sau 1 lần lặp ta đã tìm được giá trị cực trị.

## Giải bài toán

Trở lại với bài toán ban đầu, ta đã có được 2 phương pháp tối ưu hàm mất mát \\( J \\).
Ta sẽ giải bài này dùng phương pháp tối ưu Newton's method.  
Ta có hàm mất mát:
\\[ J_{(w)} = -\sum_{i=1}^{n}(y^{(i)} log(\alpha^{(i)}) + (1-\alpha^{(i)})log(1-\alpha^{(i)})) \\]
Áp dụng công thức Newton:
\\[ w_{t+1} = w_t - \mathbb{H}^{-1} \nabla _ {w} J_{(w_t)}\\]
Ta cần phải tính đạo hàm bật nhất và bật 2 của hàm mất mát trước.
\\[ log \alpha ^{(i)} = log \frac{1}{1+e^{-w^Tx^{(i)}}} = -log(1+e^{-w^Tx^{(i)}}) \\]
\\[ \frac{\partial log \alpha ^{(i)}}{\partial w_j} = \frac{x _ {j} ^ {(i)} e^{-w^Tx^{(i)}}}{1+e^{-w^Tx^{(i)}}} = x _ {j} ^{(i)} (1 - \alpha ^{(i)} ) \\]
\\[ log(1-\alpha ^{(i)}) = log \frac{e^{-w^Tx^{(i)}}}{1+e^{-w^Tx^{(i)}}} = -w^Tx^{(i)} - log(1+e^{-w^Tx^{(i)}}) \\]
\\[ \frac{\partial log(1-\alpha ^{(i)}) }{\partial w_j} = - x _ {j} ^ {(i)} + x _ {j} ^ {(i)} (1-\alpha^{(i)}) = -\alpha^{(i)}x _ {j} ^ {(i)} \\]
Ta thay vào để tính đạo hàm \\( J_{(w)} \\) ta được:  

\\[ \frac{ \partial J_{(w)} }{ \partial w_j } = - \sum_{j=1}^{n}( y^{(i)} x _ {j} ^ {(i)} (1 - \alpha ^ {(i)} ) - ( 1 - y^{(i)} ) x _ {j} ^ {(i)} \alpha ^ {(i)}  ) \\]

\\[ = - \sum _ {j=1} ^ {n} ( y^{(i)} x _ {j} ^ {(i)} - y^{(i)} x _ {j} ^{(i)} \alpha ^{(i)}  - x _ {j} ^ {(i)} \alpha ^{(i)}  + y^{(i)} x _ {j} ^ {(i)} \alpha ^{(i)}  ) \\]

\\[ = - \sum _ {j=1} ^ {n}( y^{(i)} x _ {j} ^{(i)}  - x _ {j} ^{(i)} \alpha ^{(i)}  ) =  \sum_{j=1}^{n} x _ {j} ^{(i)}( \alpha^{(i)} - y^{(i)}  ) \\]

Một cách tổng quát cho hàm nhiều biến:
\\[ \nabla _ x J = A^T ( \alpha - Y ) \\]
Với :



<!--    \\[  \\]  \\(  \\)   -->

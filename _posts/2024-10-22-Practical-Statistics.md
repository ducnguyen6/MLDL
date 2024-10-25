---
layout: post
title: Practical Statistics
mathjax: true
tags:
- Statistic
categories: Statistic
description: 
---

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) thường là việc làm đầu tiên của một data scientist khi tiếp cận với dữ liệu. Mục đích để thăm dò, phân tích sơ bộ và đánh giá dữ liệu. 

### Cấu trúc dữ liệu

Các loại dữ liệu khác nhau được lưu ở nhiều cấu trúc khác nhau để phù hợp có việc lưu trữ và tính toán.

**Dữ liệu cơ bản:**
- Dữ liệu liên tục (Continuous): biển diễn các giá trị liên tục, thường là số thực, cơ bản như *float, double, numeri*c.
- Dữ liệu rời rạc (Discrete): biểu diễn thường là số nguyên, cơ bản như *integer*.
- Dữ liệu phân loại (Categorical): cũng là dữ liệu rời rạc nhưng không có giá trị so sánh như Discrete, cơ bản như *enums*.
- Dữ liệu nhị phân (Binary): biểu diễn giá trị nhị phân (0, 1) hoặc luận lý true/false, cơ bản như *boolean*.

**Dữ liệu bảng:**
Dữ liệu dạng bảng hay Rectangular, Frame là dữ liệu được lưu trữ dạng bảng gồm các dòng và các cột. 
Phổ biến nhất là Dataframe và database table. Trong máy tính, các bảng thường được lưu giống như 1 mảng 2 chiều.

Cấu trúc cơ bản gồm 
- Columns: các cột thường được sử dụng để lưu trữ các biến cùng loại, như 1 feature cụ thể. phần tử trên một cột sẽ có cùng kiểu dữ liệu.
- Rows/Records: lưu trữ các mẫu (sample) dữ liệu.

**Dữ liệu phi bảng tính:**

Ngoài dữ liệu dạng bảng (Rectagular), các dữ liệu có thể lưu trữ ở dạng phi bảng tính (Nonrectangular Data Structures) để phục vụ cho việc tính toán và truy vấn phức tạp hơn hoặc phù hợp với bài toán cụ thể. Các dữ liệu phi bảng tính như *Time-series, Graph*.

### Ước tính vị trí - Estimates of Location

Các biến trong dữ liệu có thể có rất nhiều giá trị, việc ước tính vị trí dữ liệu cho chúng ta biết được các giá trị điển hình của dữ liệu đó. Các phương pháp ước tính vị trí gồm:
- Mean: Giá trị trung bình của dữ liệu. Đây là giá trị cơ bản nhất và được dùng nhiều nhất để hiểu cơ bản về dữ liệu.
- Median: Giá trị trung vị, là giá trị nằm giữa của dữ liệu, có một nữa dữ liệu lớn hơn và 1 nữa dữ liệu nhỏ hơn. Giá trị median này đặc biệt dùng cho các bài toán khi có các phần tử quá lớn hoặc quá nhỏ khiến cho việc tính trung bình bị ảnh hưởng bởi các phần tử đó.
- Mode: giá trị xuất hiện nhiều nhất trong tập dữ liệu, giá trị này đặc biệt dùng cho kiểu dữ liệu rời rạc hay phân loại (categorical).
- Outlier: Các giá trị ngoại lệ, rất khác so với các giá trị còn lại trong tập dữ liệu.

### Ước tính về sự thay đổi - Estimates of Variability

Ước tính vị trí (Estimates of Location) chỉ cho ta thông tin về 1 giá trị đặc trưng cho tập dữ liệu, ngoài ra chúng ta cũng muốn hiểu hơn về sự biến thiên, phân tán của các dữ liệu trong tập dữ liệu. 

Các đặc tính thay đổi của dữ liệu thường có là:
- Mean Absolute Deviation - MAD (trung bình độ lệch tuyệt đối): được đo bằng trung bình của các giá trị tuyệt đối của hiệu giá trị trung bình (mean) và giá trị của điểm dữ liệu. nó đơn giản để tính xem trung bình các điểm dữ liệu lệch bao nhiêu so với giá trị trung bình.
- Variance (phương sai): Phương sai hay bình phương sai số được tính tương tự như MAD, tuy nhiên thay vì tính trị tuyệt đối của hiệu thì sẽ tính bình phương của hiệu. Việc tính bình phương cũng đảm bảo giá trị là không âm và các lệch nhiều thì giá trị của phương sai càng lớn do tính chất của việc tính bình phương.
- Standard Deviation: Độ lệch chuẩn được tính bằng căn bậc 2 của phương sai, độ lệch chuẩn dùng để đo mức độ phân tán của một tập hợp dữ liệu so với giá trị trung bình của nó. Nói cách khác, độ lệch chuẩn cho biết các số liệu trong tập dữ liệu có phân bố gần nhau hay xa nhau.Cũng gần giống như MAD, phương sai được dùng nhiều hơn trong thống kê.
- Percentile: còn gọi là bách phân vị, dùng để chỉ vị trí tương đối của một giá trị so với các giá trị còn lại trong một tập dữ liệu. Nói cách khác, percentile cho biết có bao nhiêu phần trăm dữ liệu có giá trị nhỏ hơn hoặc bằng giá trị đang xét. Median sẽ là Percentile 50 (50% dữ liệu có giá trị nhỏ hơn nó). Quartile hay tứ phân vị là một loại Percentile phổ biến thường được dùng, chỉ dữ liệu ở các mốc 25%, 50% và 75%.

###  Data Distribution

**Boxp lot**

![Box plot](/MLDL/assets/img/boxplot.png)

Boxplot là một cách biểu diễn phân bố dư liệu đơn giản nhưng rất hiệu quá dựa trên phân vị.

Box là một hình chữ nhật thể 2 mốc phân vị Q75 và Q25. Trung tâm hình chữ nhật có đánh dấu vị trí trung vị (median) hay Q50.

Có 2 râu tại maximun và minimun là 2 giá trị được xem là giới hạn của các dữ liệu bình thường. Các điểm dữ liệu nằm ngoài 2 điểm này được xem là outliers.

Cách tính 2 râu này có nhiều cách, có thể là Q5 và Q95, hoặc có thể tính bằng khoảng cách từ nó đến Q25 và Q75 là 1.5*IQR, với IQR (interquartile range) là khoảng cách từ Q75 - Q25 (chiều dài của hình chữ nhật).

Boxplot thể hiện được mức độ phân bố của dữ liệu, độ lệch của phân bố và giới hạn outliers.

**Histogram**

![Histogram plot](/MLDL/assets/img/histplot.png)

Histogram là một cách biểu diển cổ điển của phân bố dữ liệu, giá trị của dữ liệu được thể hiện bằng các cột rời rạc, với chiều cao của các cột thể hiện số lượng của các điểm dữ liệu có giá trị bằng giá trị cột đó đại diện.

**Density plot**

![Density plot](/MLDL/assets/img/displot.png)

Density plot hay distribution plot là cách biểu diễn tương tự với Histogram, tuy nhiên density plot vẽ các đường cong thể hiện mực độ phân bố của dữ liệu thay vì biển diển bằng các cột rời rạc. Trong hình mô tả, đường cong được đè lên trên histogram là density plot của phân bố.

### Khám phá Binary và Categorical Data

Đối với dữ liệu Binary và Categorical, chỉ số thống kê cơ bản Mode, thể hiện giá trị nào có số lượng phổ biến nhất trong tập dữ liệu.

Các loại biểu đồ thường được sử dụng để biểu diễn phân bố của Binary và Categorical Data là *Bar charts*, *Pie charts*.

### Correlation

**Correlation coefficient**
Hệ số tương quan (Correlation coefficient) là thước đo thống kê cho biết mức độ liên hệ tuyến tính giữa 2 biến số. Dựa vào chỉ số này có thể xem 2 biến có xu hướng thay đổi cùng chiều hay ngược chiều, mức độ thay đổi mạnh hay yếu.

Có các loại hệ số phổ biến:
- Pearson: 
- Spearman:
- Kendall:

**Correlation matrix**

**Scatterplot**


### Exploring Two or More Variables

## Data and Sampling Distributions

### Random Sampling and Sample Bias


## Statistical Experiments and Significance Testing


## Regression and Prediction

## Classification

## Statistical Machine Learning

## Unsupervised Learning

## nn.Conv1d层的输出形状

对于一个一维卷积层，其输出长度（Output Length, $l_o$）的计算公式如下：

$l_o = \left\lfloor\frac{l_i + 2 \times p - f}{s} + 1\right\rfloor$

其中：

* $l_i$：输入数据的长度（比如，一段音频有100个时间步，$l_i$ = 100）。

* $f$：卷积核的大小（比如，一个长度为3的卷积核，$f$ = 3）。

* $s$：卷积核每次移动的步长，默认为 1。

* $p$：在输入数据两端填充的零的个数，默认为0。

注意：⌊ ... ⌋ 表示向下取整（floor操作）。

## nn.MaxPool1d输出形状

nn.MaxPool1d输出长度（Output Length, $l_o$）的计算公式如下：

$l_o = \left\lfloor\frac{l_i + 2 \times p - f}{s} + 1\right\rfloor$

其中：

$l_i$：输入数据的长度。

$f$：池化窗口的大小。

$s$：池化窗口每次移动的步长。默认为$f$。这是一个与nn.Conv1d (默认1)的重要区别。

$p$：在输入数据两端填充的零的个数，默认为 0。


注意：⌊ ... ⌋ 表示向下取整（floor操作）。
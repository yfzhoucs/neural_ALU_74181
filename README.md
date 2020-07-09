# neural_ALU_74181
ALU 74181 implemented by Neural Networks with 100% accuracy.

## ALU 74181
> The 74181 is a 4-bit slice arithmetic logic unit (ALU).  
> ...  
> It was used as the arithmetic/logic core in the CPUs of many historically significant minicomputers and other devices.  

[quote source](https://en.wikipedia.org/wiki/74181)

![Arithmetic Logic Circuit of 74181](/img/74181aluschematic.png)  
[img source](https://en.wikipedia.org/wiki/74181)

## How to run Neural 74181
Run 
```
python alu_74181.py
```
The model has been pretrained and the ckpts are included in this repo.  
The pytorch model is defined in alu_74181.py. A simple running example is also included there. You can easily modify the input signals and see what would happen.

## Accuracy of this Neural Network model
In the pretrained ckpts, floats between \[0, 0.1) are viewed as LOW and 0, while floats between (0.9, 1\] are viewed as HIGH and 1.  
With this standard, the model achieves 100% accuracy.

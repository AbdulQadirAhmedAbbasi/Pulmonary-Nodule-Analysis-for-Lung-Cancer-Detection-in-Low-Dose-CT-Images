Code Flow:

1) Start with "NodulePatchExtraction.py". This code will create 3D Patches & Labels (Only Noduels)

2) Then go with "NonNodulePatchExtraction.py". This code will create 3D Patches & Labels (Only Non Noduels) 

3) Followed by "LabelCreartion(Nodules+NOnNodules).py". This code will create a single CSV file having both Nodule & NonNodule Labels

4) In Last use "3DCNNArchitecture.py". This will first prepare train and test data, then model will be defined and executed with result storage.


Personal Comments:

1) Anything called "Positive" in code is reffered to "Nodule" & "Negative" to "NonNodule".
2) Patch Size is 32*32*32 (can be changed and will demand minute changes in codes)
3) For Model Clarification, kindly see the Research Paper in this folder
4) Kindly change paths given in code for different CSV files & DataSet according to thier presence in your Local Machine  


Data Set: 

1) Provided data is not ready to be used directly for 3D CNN
2) We only have 888 CT scans in .mhd format
3) We have annotation of only nodules with different diameters
4) Data preprocessing requires Patch Extraction + Label Creation for both Nodules & Non Nodules 
5) All preprocessed dataset is present on Dr.Hafeez's System at National University of Computer & Emerging Sciences.

For Questions:

Email: ehmadabbasi@gmail.com
Mob: +923134238087 

Regards,
Abdul Qadir Ahmed Abbasi
 

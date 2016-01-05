% Gesture detection using wavelet transform and Support Vector Machine

% Code by Jishnu, Mahesh M, Vivek Ashokan
% Date : 19/10/2015

%initialize workspace
clc;
clear all;
close all;

% train the SVM
width=200;height=300; %Set width,height



% set source for training images 
trainIm = dir('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Train_Images\*.jpg');  
trainCount = length(trainIm);
train = [];

%process each image and get the wavelet coefficients for training vector 

%-------------------------------------------------------------------------------------------------------------------------
%Start of 1st binary classification for MultiSVM 
%MultiSVM Compare first two sets.

for i = 1 : ceil(2*trainCount/3)
    filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Train_Images\',trainIm(i).name);
    I = imread(filename);
    %figure,imshow(I);
    
    % Resize & Convert to grayscale image
    I=imresize(I,[height width]);
    I=rgb2gray(I);
    %figure,imshow(I);
    
    % Converting to binary image
    I=im2bw(I,0.46);
    %figure,imshow(I);
    
    % structural element for morphological operations
    se = strel('disk',11);  
    % morphological operations
    I1 = imerode(I,se);
    I2 = imdilate(I1,se);
    
       
    %For V1.0 code uncomment below lines
%     % steps to find the number of levels required in wavelet transform
%     
%     % size of image
%     sizeim = size(I2);
% 
%     % number of levels
%     if(sizeim(1) > sizeim(2))
%         level = floor(log2(sizeim(1)));
%     else
%         level = floor(log2(sizeim(2)));
%     end
    
    %for V1.0 code uncomment upto above code
    

    
    %decompositions levels = 7
    level = 7;

    [c,s] = wavedec2(I2,level,'db1');
    
    detCoeff = detcoef2('compact',c,s,level);
    
    train = vertcat(train,detCoeff);  
    
end


% initialize group with zeros
 train_label = zeros(ceil(2*trainCount/3),1);

% assuming training images in order, make first half rows +1 group and 2nd half rows -1 group
 train_label(1:floor(trainCount/3)) = 1; %All the best
 train_label(floor((trainCount/3)+1):floor(2*(trainCount/3))) = -1; %Victory
 

% Perform first run of svm training
SVMStruct = svmtrain(train , train_label, 'kernel_function', 'linear');


% set source for testing images 
testIm = dir('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\*.jpg');  
testCount = length(testIm);
test = [];

%process each image and get the wavelet coefficients for training vector

%MultiSVM classification variables
for i=1:testCount
A(i)=0;
V(i)=0;
H(i)=0;
end


for i = 1:testCount
    
 filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\',testIm(i).name);
    I = imread(filename);
    
%Converting to grayscale image
I=imresize(I,[height width]);
I=rgb2gray(I);
% figure,imshow(I);

% Converting to binary image
I=im2bw(I,0.46);
%figure,imshow(I);

% structural element for morphological operation
se = strel('disk',11);  
% morphological operation
I1 = imerode(I,se);
I2 = imdilate(I1,se);

level = 7;
[c,s] = wavedec2(I2,level,'db1');
detCoeff = detcoef2('compact',c,s,level);
test = vertcat(test,detCoeff);  

end

%Gesture Classification -> Assigning groups to test images 
Group = svmclassify(SVMStruct,test);

for i = 1:testCount
    
if(Group(i) == 1)
fprintf('\n%s : All The Best',testIm(i).name);
A(i)=A(i)+1;
end
if(Group(i) == -1)
   fprintf('\n%s : Victory',testIm(i).name);
   V(i)=V(i)+1;
end
if(Group(i) == 0)
   fprintf('\n%s : Hifive',testIm(i).name);
   H(i)=H(i)+1;
end
       
end

%Debug Code -->  Image Display with Gesture - 1st Stage

% for i = 1:testCount
%     
% if(Group(i) == 1)
%     subplot(ceil(sqrt(testCount)),ceil(sqrt(testCount)),i);
%     filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\',testIm(i).name);
%     I=imread(filename);
%      I=imresize(I,[height width]);
%      imshow(I);
%     title(sprintf('%s : A\n',testIm(i).name));
% 
% end
% if(Group(i) == -1)
%      subplot(ceil(sqrt(testCount)),ceil(sqrt(testCount)),i);
%      filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\',testIm(i).name);
%      I=imread(filename);
%      I=imresize(I,[height width]);
%      imshow(I);
%       title(sprintf('%s : V\n',testIm(i).name));
% end
%  if(Group(i) == 0)
%      subplot(ceil(sqrt(testCount)),ceil(sqrt(testCount)),i);
%      filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\',testIm(i).name);
%      I=imread(filename);
%      I=imresize(I,[height width]);
%      imshow(I);
%       title(sprintf('%s : H\n',testIm(i).name));
% end      
% end



fprintf('\n Classification after 1st stage of MultiSVM \n');
fprintf('All the Best : %d \n',A);
fprintf('Victory : %d \n',V);
fprintf('Hifive  : %d \n',H);

%End of 1st binary classification for MultiSVM
%-------------------------------------------------------------------------------------------------------------------------




%-------------------------------------------------------------------------------------------------------------------------
%Start of 2nd binary classification for MultiSVM

%Resetting the train set
train = [];

%MultiSVM Compare last two sets.
for i = ceil((trainCount/3)+1) : ceil(trainCount)
    filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Train_Images\',trainIm(i).name);
    I = imread(filename);
    %figure,imshow(I);
    
    % Resize & Convert to grayscale image
    I=imresize(I,[height width]);
    I=rgb2gray(I);
    %figure,imshow(I);
    
    % Converting to binary image
    I=im2bw(I,0.46);
    %figure,imshow(I);
    
    % structural element for morphological operations
    se = strel('disk',11);  
    % morphological operations
    I1 = imerode(I,se);
    I2 = imdilate(I1,se);
    %figure,imshow(erodedBW);
    
    level = 7;
    [c,s] = wavedec2(I2,level,'db1');
    detCoeff = detcoef2('compact',c,s,level);
    train = vertcat(train,detCoeff);  
    
end

   
train_label               = zeros(size(30,1),1);
train_label(1:15,1)   = -1;          % -1 = Victory
train_label(16:30,1)  =  0;          %  0 = Hifive

% Perform first run of svm
SVMStruct = svmtrain(train , train_label, 'kernel_function', 'linear');

% set source for training images 
testIm = dir('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\*.jpg');  
testCount = length(testIm);
test = [];

%process each image and get the wavelet coefficients for training vector

for i = 1:testCount
    
 filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\',testIm(i).name);
    I = imread(filename);
    
%Converting to grayscale image
I=imresize(I,[height width]);
I=rgb2gray(I);
% figure,imshow(I);

% Converting to binary image
I=im2bw(I,0.46);
%figure,imshow(I);

% structural element for morphological operation
se = strel('disk',11);  
% morphological operation
I1 = imerode(I,se);
I2 = imdilate(I1,se);


 level = 7;
 [c,s] = wavedec2(I2,level,'db1');
 detCoeff = detcoef2('compact',c,s,level);
 test = vertcat(test,detCoeff);  

end

Group = svmclassify(SVMStruct,test);

for i = 1:testCount
    
if(Group(i) == 1)
fprintf('\n%s : All The Best',testIm(i).name);
A(i)=A(i)+1;
end
if(Group(i) == -1)
   fprintf('\n%s : Victory',testIm(i).name);
   V(i)=V(i)+1;
end
if(Group(i) == 0)
   fprintf('\n%s : Hifive',testIm(i).name);
   H(i)=H(i)+1;
end
       
end

%Debug Code -->  Image Display with Gesture - 2nd Stage

% for i = 1:testCount
%     
% if(Group(i) == 1)
%     subplot(ceil(sqrt(testCount)),ceil(sqrt(testCount)),i);
%     filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\',testIm(i).name);
%     I=imread(filename);
%      I=imresize(I,[height width]);
%      imshow(I);
%     title(sprintf('%s : A\n',testIm(i).name));
% 
% end
% if(Group(i) == -1)
%      subplot(ceil(sqrt(testCount)),ceil(sqrt(testCount)),i);
%      filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\',testIm(i).name);
%      I=imread(filename);
%      I=imresize(I,[height width]);
%      imshow(I);
%       title(sprintf('%s : V\n',testIm(i).name));
% end
%  if(Group(i) == 0)
%      subplot(ceil(sqrt(testCount)),ceil(sqrt(testCount)),i);
%      filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\',testIm(i).name);
%      I=imread(filename);
%      I=imresize(I,[height width]);
%      imshow(I);
%       title(sprintf('%s : H\n',testIm(i).name));
% end      
% end


fprintf('\n Classification after 2nd stage of MultiSVM \n');
fprintf('All the Best : %d \n',A);
fprintf('Victory : %d \n',V);
fprintf('Hifive  : %d \n',H);

%End of 2st binary classification for MultiSVM
%-------------------------------------------------------------------------------------------------------------------------



%-------------------------------------------------------------------------------------------------------------------------
%Start of 3rd binary classification for MultiSVM

%Resetting the train set
train = [];

% MultiSVM Compare first and third set 
for i = 1:ceil(trainCount/3)   
    %   if(trainCount <= ceil(trainCount/3) | trainCount >ceil(2*trainCount/3))
    filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Train_Images\',trainIm(i).name);
    I = imread(filename);
    %figure,imshow(I);
    
    % Resize & Convert to grayscale image
    I=imresize(I,[height width]);
    I=rgb2gray(I);
    %figure,imshow(I);
    
    % Converting to binary image
    I=im2bw(I,0.46);
    %figure,imshow(I);
    
    % structural element for morphological operations
    se = strel('disk',11);  
    % morphological operations
    I1 = imerode(I,se);
    I2 = imdilate(I1,se);

     level = 7;
     [c,s] = wavedec2(I2,level,'db1');
     detCoeff = detcoef2('compact',c,s,level);
     train = vertcat(train,detCoeff);  
  

end

  for i = ceil((2*trainCount/3)+1):trainCount

    filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Train_Images\',trainIm(i).name);
    I = imread(filename);
    %figure,imshow(I);
    
    % Resize & Convert to grayscale image
    I=imresize(I,[height width]);
    I=rgb2gray(I);
    %figure,imshow(I);
    
    % Converting to binary image
    I=im2bw(I,0.46);
    %figure,imshow(I);
    
    % structural element for morphological operations
    se = strel('disk',11);  
    % morphological operations
    I1 = imerode(I,se);
    I2 = imdilate(I1,se);
    
    level = 7;
    [c,s] = wavedec2(I2,level,'db1');
    detCoeff = detcoef2('compact',c,s,level);
    train = vertcat(train,detCoeff);  
  

end


 train_label               = zeros(size(30,1),1);
 train_label(1:15,1)   = 1;          %  1 = All the best
 train_label(16:30,1)  = 0;          %  0 = Hifive

 
SVMStruct = svmtrain(train , train_label, 'kernel_function', 'linear');


% set source for training images 
testIm = dir('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\*.jpg');  
testCount = length(testIm);
test = [];

%process each image and get the wavelet coefficients for training vector



for i = 1:testCount
    
 filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\',testIm(i).name);
    I = imread(filename);
    
%Converting to grayscale image
I=imresize(I,[height width]);
I=rgb2gray(I);
% figure,imshow(I);

% Converting to binary image
I=im2bw(I,0.46);
%figure,imshow(I);

% structural element for morphological operation
se = strel('disk',11);  
% morphological operation
I1 = imerode(I,se);
I2 = imdilate(I1,se);


    level = 7;
    [c,s] = wavedec2(I2,level,'db1');
    detCoeff = detcoef2('compact',c,s,level);
    test = vertcat(test,detCoeff);  


end

Group = svmclassify(SVMStruct,test);

for i = 1:testCount
    
if(Group(i) == 1)
fprintf('\n%s : All The Best',testIm(i).name);
A(i)=A(i)+1;
end
if(Group(i) == -1)
   fprintf('\n%s : Victory',testIm(i).name);
   V(i)=V(i)+1;
end
if(Group(i) == 0)
   fprintf('\n%s : Hifive',testIm(i).name);
   H(i)=H(i)+1;
end
       
end


%Debug Code -->  Image Display with Gesture - 3rd Stage

% for i = 1:testCount
%     
% if(Group(i) == 1)
%      
% 
% end
% if(Group(i) == -1)
%      subplot(ceil(sqrt(testCount)),ceil(sqrt(testCount)),i);
%      filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\',testIm(i).name);
%      I=imread(filename);
%      I=imresize(I,[height width]);
%      imshow(I);
%       title(sprintf('%s : V\n',testIm(i).name));
% end
%  if(Group(i) == 0)
%      subplot(ceil(sqrt(testCount)),ceil(sqrt(testCount)),i);
%      filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\',testIm(i).name);
%      I=imread(filename);
%      I=imresize(I,[height width]);
%      imshow(I);
%       title(sprintf('%s : H\n',testIm(i).name));
% end      
% end



fprintf('\n Classification after 3rd stage of MultiSVM \n');
fprintf('All the Best : %d \n',A);
fprintf('Victory : %d \n',V);
fprintf('Hifive  : %d \n',H);

%End of 3st binary classification for MultiSVM
%-------------------------------------------------------------------------------------------------------------------------


%Final Image Classification - Classification based on information from 3
%sets of binary classification

for i=1:testCount
if A(i)==2
  fprintf('\n Image %s denotes the gesture -> All the Best\n',testIm(i).name);
   subplot(ceil(sqrt(testCount)),ceil(sqrt(testCount)),i);
     filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\',testIm(i).name);
     I=imread(filename);
     I=imresize(I,[height width]);
     imshow(I);
     title(sprintf('%s -> All the Best\n',testIm(i).name));
elseif V(i)==2
    fprintf('\n Image %s denotes the gesture -> Victory \n',testIm(i).name);
 subplot(ceil(sqrt(testCount)),ceil(sqrt(testCount)),i);
     filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\',testIm(i).name);
     I=imread(filename);
     I=imresize(I,[height width]);
     imshow(I);
    title(sprintf('%s -> Victory\n',testIm(i).name));
elseif H(i)==2
    fprintf('\n Image %s denotes the gesture -> Hifive\n',testIm(i).name);
     subplot(ceil(sqrt(testCount)),ceil(sqrt(testCount)),i);
     filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\',testIm(i).name);
     I=imread(filename);
     I=imresize(I,[height width]);
     imshow(I);
      title(sprintf('%s -> Hifive\n',testIm(i).name));
      else    
    fprintf('\n Image %s denotes the gesture -> Invalid\n',testIm(i).name);
     subplot(ceil(sqrt(testCount)),ceil(sqrt(testCount)),i);
     filename = strcat('C:\Users\Vivek\Downloads\Documents\Mtech\DSP_proj\DSP_Term_Paper\Test_Images\',testIm(i).name);
     I=imread(filename);
     I=imresize(I,[height width]);
     imshow(I);
      title(sprintf('%s -> Invalid\n',testIm(i).name));
end

end

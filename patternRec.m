clc; clear;
minmax_norm = @(X) (X-min(min(X)))/(max(max(X))-min(min(X)));

imagefiles = dir('Data\Resim*'); 
mask  = imread('oiltank.jpg');
%% make template image gray and find sizes
g_mask = im2double(rgb2gray(mask));
[maskrow,maskcol] = size(g_mask);

halfrow = fix(maskrow/2);
halfcol = fix(maskcol/2);

%% rotate template and normalize
gr_mask = rot90(g_mask, 2);
nr_mask = normalize(gr_mask);
%% not rotated for correlation
n_mask = normalize(g_mask);

for index = 1:8
     currentfilename = imagefiles(index).name;
    input = imread("Data/"+currentfilename);
    
    %% make input image gray and normalize
    g_input = im2double(rgb2gray(input));
    [in_row,in_col] = size(g_input);
    n_input = normalize(g_input);
    
    %% convolution of input and template
    conv2r = conv2(n_input,nr_mask);
    myConv2r = myConv2(n_input,nr_mask);

    %% correlation results of input and template
    corr2r = xcorr2(n_input,n_mask);

    %% get min-max normalization of results
    conv2Result = minmax_norm(conv2r);
    myConv2Result = minmax_norm(myConv2r);
    corr2Result = minmax_norm(corr2r);
    %%
    figure(index); set(gcf, 'Position', get(0, 'Screensize')); subplot(1,3,1);
    imshow(input); 
    title(currentfilename+" conv2 sonucu"); hold on;

    figure(index); subplot(1,3,2);
    imshow(input); 
    title(currentfilename+" myConv2 sonucu"); hold on;

    figure(index); subplot(1,3,3);
    imshow(input); 
    title(currentfilename+" xcorr2 sonucu"); hold on;
    
     %% get the location of max of the result
        i = 1;
        [max1(i),imax1] = max(conv2Result(:));
        [max_row1,max_col1] = ind2sub(size(conv2Result),imax1);

    while max1(i)>0.92  %treshold
        %% if the founded point is at edge
        if max_row1<halfrow
            max_row1=halfrow;
        end
        if max_col1<halfcol
            max_col1=halfcol;
        end
        
        %% paint the area black
        for j=1:maskrow
            for k=1:maskcol
                conv2Result(max_row1-halfrow+j,max_col1-halfcol+k) = 0;
            end
        end
        %% draw a rectangle    
        figure(index),subplot(1,3,1),rectangle('Position', [max_col1-maskcol max_row1-maskrow maskcol maskrow], ...
                  'EdgeColor', [0.5 0 0], ...
                  'LineWidth', 3, ...
                  'LineStyle', '-');
    i = i+1;
    %% get the location of max of the result
        [max1(i),imax1] = max(conv2Result(:));
        [max_row1,max_col1] = ind2sub(size(conv2Result),imax1);
    end
    text(10,in_row+30,"conv2 fonksiyonuyla "+(i-1)+ " yağ tankı bulundu") 

     %% get the location of max of the result
        i = 1;
        [max2(i),imax2] = max(myConv2Result(:));
        [max_row2,max_col2] = ind2sub(size(myConv2Result),imax2);
    % myConv2 results
    while max2(i)>0.92  %treshold
        %% if the founded point is at edge
        if max_row2<halfrow
            max_row2=halfrow;
        end
        if max_col2<halfcol
            max_col2=halfcol;
        end
        
        %% paint the area black
        for j=1:maskrow
            for k=1:maskcol
                myConv2Result(max_row2-halfrow+j,max_col2-halfcol+k) = 0;
            end
        end
        %% draw a rectangle    
        figure(index),subplot(1,3,2),rectangle('Position', [max_col2-maskcol max_row2-maskrow maskcol maskrow], ...
                  'EdgeColor', [0.5 0 0], ...
                  'LineWidth', 3, ...
                  'LineStyle', '-');
    i = i+1;
    %% get the location of max of the result
        [max2(i),imax2] = max(myConv2Result(:));
        [max_row2,max_col2] = ind2sub(size(myConv2Result),imax2);
    end
    text(10,in_row+30,"myConv2 fonksiyonuyla "+(i-1)+ " yağ tankı bulundu") 

    %% get the location of max of the result
        i = 1;
        [max3(i),imax3] = max(corr2Result(:));
        [max_row3,max_col3] = ind2sub(size(corr2Result),imax3);
    % myConv2 results
    while max3(i)>0.92  %treshold
        %% if the founded point is at edge
        if max_row3<halfrow
            max_row3=halfrow;
        end
        if max_col3<halfcol
            max_col3=halfcol;
        end
        
        %% paint the area black
        for j=1:maskrow
            for k=1:maskcol
                corr2Result(max_row3-halfrow+j,max_col3-halfcol+k) = 0;
            end
        end
        %% draw a rectangle    
        figure(index),subplot(1,3,3),rectangle('Position', [max_col3-maskcol max_row3-maskrow maskcol maskrow], ...
                  'EdgeColor', [0.5 0 0], ...
                  'LineWidth', 3, ...
                  'LineStyle', '-');
    i = i+1;
    %% get the location of max of the result
        [max3(i),imax3] = max(corr2Result(:));
        [max_row3,max_col3] = ind2sub(size(corr2Result),imax3);
    end
    text(10,in_row+30,"xcorr2 fonksiyonuyla "+(i-1)+ " yağ tankı bulundu") 
end



function res = myConv2(A, B) 
    [Ax,Ay] = size(A);
    [Bx,By] = size(B);
    B = rot90(B, 2);

    padded = zeros(Ax + 2*(Bx-1), Ay + 2*(By-1));
    for i = Bx : Bx+Ax-1
        for j = By : By+Ax-1
            padded(i,j) = A(i-Bx+1, j-By+1);
        end
    end

    res = zeros(Ax+Bx-1,Ay+By-1);
    for i = 1 : Ax+Bx-1
        for j = 1 : By+Ay-1
            for k = 1 : Bx
                for m = 1 : By
                    res(i, j) = res(i, j) + (padded(i+k-1, j+m-1) * B(k, m));
                end
            end
        end
    end
end

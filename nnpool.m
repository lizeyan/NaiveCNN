function output = nnpool(input, kernel_size, pad)
    % Your codes here     
    % hint:
    %     1. pad zeros on the input's each side
    %     2. use im2col to extract consecutive kernel_size * kernel_size
    %        patches from input
    %     3. get average value in each patch
    %     4. restore the original layout
    [h, w, c, n] = size (input);
    hn = h + pad * 2;
    wn = w + pad * 2;
    output = zeros(hn / kernel_size, wn / kernel_size, c, n);
    input_shape = zeros(hn, wn, c, n);
    input_shape (pad + 1:pad + h, pad + 1:pad + w, :, :) = input;
    for n_cnt = 1:n
        for c_cnt = 1:c
%             tic
%             col = im2col (input_shape(:, :, c_cnt, n_cnt), [kernel_size, kernel_size], 'distinct');
%             toc
%             tic
            col = im2col_distinct (input_shape(:, :, c_cnt, n_cnt), [kernel_size, kernel_size]);
%             toc
            col = mean (col);
%             disp (size(col));
%             output (:, :, c_cnt, n_cnt) = col2im (col, [1, 1], [hn / kernel_size, wn / kernel_size], 'distinct');
            output(:, :, c_cnt, n_cnt) = reshape (col, [hn / kernel_size, wn / kernel_size]);
%             disp (isequal(test, output(:, :, c_cnt, n_cnt)));
%             disp (size (output(:, :, c_cnt, n_cnt)));
        end
    end
%     disp (size(input));
%     disp (size(input_shape));
%     disp (size(output));
end

function out = im2col_distinct(A,blocksize)

    nrows = blocksize(1);
    ncols = blocksize(2);
    nele = nrows*ncols;

    row_ext = mod(size(A,1),nrows);
    col_ext = mod(size(A,2),ncols);

    padrowlen = (row_ext~=0)*(nrows - row_ext);
    padcollen = (col_ext~=0)*(ncols - col_ext);

    A1 = zeros(size(A,1)+padrowlen,size(A,2)+padcollen);
    A1(1:size(A,1),1:size(A,2)) = A;

    t1 = reshape(A1,nrows,size(A1,1)/nrows,[]);
    t2 = reshape(permute(t1,[1 3 2]),size(t1,1)*size(t1,3),[]);
    t3 =  permute(reshape(t2,nele,size(t2,1)/nele,[]),[1 3 2]);
    out = reshape(t3,nele,[]);

end
            
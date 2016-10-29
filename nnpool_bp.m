function down_delta = nnpool_bp(input, delta, kernel_size, pad)
    % Your codes here
    % hint:
    %     follow the formula in slide (page 19)
    %     Generally speaking, the delta from upper layer is upsampled
    %     averagely to the down_delta in each pooling kernel_size
    [h, w, c, n] = size (input);
    down_delta = zeros(h, w, c, n);
    for n_cnt = 1:n
        for c_cnt = 1:c
%             col = im2col_distinct (delta(:, :, c_cnt, n_cnt), [1, 1]);
%             col = repmat (col, [kernel_size * kernel_size, 1]);
%             im = col2im (col, [kernel_size, kernel_size], [h + pad * 2, w + pad * 2], 'distinct');
%             down_delta(:, :, c_cnt, n_cnt) = im (pad + 1: pad + h, pad + 1: pad + w);
            down_delta(:, :, c_cnt, n_cnt) = kron (delta(:, :, c_cnt, n_cnt), ones(kernel_size, kernel_size));
%             disp (isequal(down_delta(:, :, c_cnt, n_cnt), test));
        end
    end
    down_delta = down_delta ./ (kernel_size * kernel_size);
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
      
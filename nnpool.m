function output = nnpool(input, kernel_size, pad)
    % Your codes here     
    % hint:
    %     1. pad zeros on the input's each side
    %     2. use im2col to extract consecutive kernel_size * kernel_size
    %        patches from input
    %     3. get average value in each patch
    %     4. restore the original layout
    [h w c n] = size (input);
    hn = h + pad * 2;
    wn = w + pad * 2;
    output = zeros(hn / kernel_size, wn / kernel_size, c, n);
    input_shape = zeros(hn, wn, c, n);
    input_shape (pad + 1:pad + h, pad + 1:pad + w, :, :) = input;
    for n_cnt = 1:n
        for c_cnt = 1:c
            col = im2col (input_shape(:, :, c_cnt, n_cnt), [kernel_size, kernel_size], 'distinct');
            col = mean (col);
%             disp (size(col));
            output (:, :, c_cnt, n_cnt) = col2im (col, [1, 1], [hn / kernel_size, wn / kernel_size], 'distinct');
%             disp (size (output(:, :, c_cnt, n_cnt)));
        end
    end
%     disp (size(input));
%     disp (size(input_shape));
%     disp (size(output));
end
            
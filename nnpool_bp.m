function down_delta = nnpool_bp(input, delta, kernel_size, pad)
    % Your codes here
    % hint:
    %     follow the formula in slide (page 19)
    %     Generally speaking, the delta from upper layer is upsampled
    %     averagely to the down_delta in each pooling kernel_size
    [h w c n] = size (input);
    down_delta = zeros(h, w, c, n);
    for n_cnt = 1:n
        for c_cnt = 1:c
            col = im2col (delta(:, :, c, n), [1, 1], 'distinct');
            col = repmat (col, [4, 1]);
            im = col2im (col, [kernel_size, kernel_size], [h + pad * 2, w + pad * 2], 'distinct');
            down_delta(:, :, c, n) = im (pad + 1: pad + h, pad + 1: pad + w);
        end
    end
end
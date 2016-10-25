function output = nnconv(input, kernel_size, num_output, W, b, pad)
    % Your codes here
    % hint:
    %     1. first pad zeros on the input's each side
    %     2. convolve input with W
    %           notice the output of j-th filter in W convolved with inpuint
    %           correspond to the j-th channel in output
    %     3. don't forget adding bias
    %
    % ps: there are more than one way in step 2, try to find the fastest method
    input_shape = zeros (size(input, 1) + pad * 2, size(input ,2) + pad * 2, size(input, 3), size(input, 4));
    input_shape(pad + 1: pad + size(input, 1), pad + 1:pad + size(input, 2), :, :) = input;
    output = zeros(size(input_shape, 1) - kernel_size + 1, size(input_shape, 2) - kernel_size + 1, num_output, size(input_shape, 4));
%     test = zeros(size(output));
%     [w_input, h_input, c_input, n_input] = size(input_shape);
%     input_cell = mat2cell (input_shape, w_input, h_input, c_input, ones(1, n_input));
%     input_cell = reshape (input_cell, [1, n_input]);
%     output_cell = cellfun(@(ic) work_case(num_output, ic, W, size(output(:, :, :, 1))), input_cell, 'UniformOutput', false);
%     output = cat (1, output_cell(:));
    for n = 1:size(input_shape, 4)
        for f = 1:num_output
            for ch = 1:size(input_shape, 3)
                output (:, :, f, n) = output(:, :, f, n) + conv2 (input_shape(:, :, ch, n), W(:, :, ch, f), 'valid');
            end
        end
%         output(:, :, :, n) = cell2mat(output_cell (n));
%         output(:, :, :, n) = work_case (num_output, input_shape(:, :, :, n), W, size(output(:, :, :, n)));
    end
%     disp (isequal(output, test));
end

function output_case = work_case (num_output, input_shape, W, size_output)
    output_case = zeros (size_output);
    for f = 1:num_output
        for ch = 1:size(input_shape, 3)
            output_case (:, :, f) = output_case(:, :, f) + conv2 (input_shape(:, :, ch), W(:, :, ch, f), 'valid');
        end
    end
end
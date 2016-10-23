function output = nnconv(input, kernel_size, num_output, W, b, pad)
    % Your codes here
    % hint:
    %     1. first pad zeros on the input's each side
    %     2. convolve input with W
    %           notice the output of j-th filter in W convolved with input
    %           correspond to the j-th channel in output
    %     3. don't forget adding bias
    %
    % ps: there are more than one way in step 2, try to find the fastest method
    for i = 1:pad
        input = pad_border (input);
    end
end

function output = pad_border (input)
    zero_column = input (:, 1, :, :) .* 0;
    output = [zero_column input zero_column];
    zero_row = output (1, :, :, :) .* 0;
    output = [zero_row; output; zero_row];
    disp(size(input));
    disp(size(output));
end
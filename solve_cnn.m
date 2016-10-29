function solve_cnn(model, loss, train_data, train_label, ...
                   test_data, test_label, solver)

    LOG_INFO('Start training...')
    num_input = size(train_data, 4);
    num_test_input = size(test_data, 4);

    train_label = one_hot_encoding(train_label, 10);
    test_label = one_hot_encoding(test_label, 10);

    if solver.shuffle
        indx = randperm(num_input);
    else
        indx = 1:num_input;
    end
    ptr = 1;

    total_loss = [];
    total_accuracy = [];
    loss_iter = zeros(1, solver.max_iter);

    for k = 1:solver.max_iter
        if ptr >= num_input
            if solver.shuffle
                indx = randperm(num_input);
            else
                indx = 1:num_input;
            end
            ptr = 1;
        end
        end_ptr = min(ptr + solver.batch_size - 1, num_input);
        batch_input = train_data(:, :, :, indx(ptr: end_ptr));
        batch_label = train_label(:, indx(ptr: end_ptr));
        ptr = end_ptr + 1;

        model.Forward(batch_input);
        grad_input = loss.backprop(model.Output(), batch_label);
        model.Backpropagation(grad_input);
        model.Update(solver.update);
        
        loss_current = loss.forward(model.Output(), batch_label);
        total_loss = [total_loss, loss_current];
        loss_iter(k) = mean(loss_current);
        total_accuracy = [total_accuracy, calc_accuracy(model.Output(), batch_label)];

        if mod(k, solver.display_freq) == 0
            mean_loss = mean(total_loss);
            mean_accuracy = mean(total_accuracy);

            msg = sprintf('Training iter %d, mean loss %.5f (batch loss %.5f), mean acc %.5f', ...
                          k, mean_loss, total_loss(end), mean_accuracy);
            LOG_INFO(msg);
        end

        if mod(k, solver.test_freq) == 0
            LOG_INFO('    Testing...')
            test_iters = ceil(num_test_input / solver.batch_size);
            test_loss = [];
            test_accuracy = [];
            for j = 1:test_iters
                test_ptr = (j - 1) * solver.batch_size + 1;
                test_end_ptr = min(j * solver.batch_size, num_test_input);

                test_batch_input = test_data(:, :, :, test_ptr: test_end_ptr);
                test_batch_label = test_label(:, test_ptr: test_end_ptr);

                test_output = model.Predict(test_batch_input);
                test_loss = [test_loss, loss.forward(test_output, test_batch_label)];
                test_accuracy = [test_accuracy, calc_accuracy(test_output, test_batch_label)];
            end
            test_mean_loss = mean(test_loss);
            test_mean_accuracy = mean(test_accuracy);

            msg = sprintf('    Testing iter %d, mean loss %.5f, mean acc %.5f', ...
                          k, test_mean_loss, test_mean_accuracy);
            LOG_INFO(msg);

            % clean train loss and accuracy buffer
            total_loss = [];
            total_accuracy = [];
        end

        if mod(k, solver.snapshot_freq) == 0
            params = model.Params();
            save(['model_weight_iter_' num2str(k) '.mat'], 'params');
            msg = sprintf('Snapshotting to model_weight_iter_%d.mat', k);
            LOG_INFO(msg);
        end
    end
    % plot loss-iteration
    f = figure ();
    plot (loss_iter);
    saveas (f, 'result/loss_iter.png')
    % plot 
    for i = 1:4
        data = test_data(:, :, 1, i);
        imwrite(data * 255, ['result/pic' num2str(i) '.png']);
        filt = model.filt (data);
        for k = 1:size(filt, 3)
            imwrite(filt(:, :, k) * 255, ['result/pic' num2str(i) '_' num2str(k) '.png']);
        end
    end
end

function LOG_INFO(msg)
    now = clock;
    fprintf('[%02d:%02d:%05.2f] %s\n', now(4), now(5), now(6), msg);
end

function encoding = one_hot_encoding(label, max_class)
    encoding = eye(max_class);
    encoding = encoding(:, label);
end

function accuracy = calc_accuracy(output, label)
    [~, pred_indx] = max(output);
    [~, label] = max(label);
    accuracy = sum(pred_indx == label) / size(label, 2);
end



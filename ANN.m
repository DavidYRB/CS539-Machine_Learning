target_train = zeros(length(optdigits_tra),10);
for i = 1:length(optdigits_tra)
    for j = 1:10
       if class_train(i) == j-1
            target_train(i,j) = 1;
       end
    end
end

target_test = zeros(length(optdigits_tes),10);
for i = 1:length(optdigits_tes)
    for j = 1:10
       if class_test(i) == j-1
            target_test(i,j) = 1;
       end
    end
end
function R = segmentation(I)
    [m, n] = size(I);
    E = 0.001;
    old_T = 256.0;
    new_T = mean(mean(I));

    while abs(old_T - new_T) > E
        sum1 = 0;
        sum2 = 0;
        count=0;
        for i=1:m
            for j = 1:n
                if I(i,j) <= new_T
                    sum1 = sum1+I(i,j);
                    count=count+1;
                else
                    sum2 = sum2+I(i,j);
                end
            end
        end
        old_T = new_T;
        new_T = (sum1/count + sum2/(m*n-count)) / 2;
    end
   
    R = zeros(m, n);
    for i = 1:m
        for j = 1:n
            if I(i, j) > new_T
                R(i, j) = 1;
            else
                R(i, j) = 0;
            end
        end
    end
end

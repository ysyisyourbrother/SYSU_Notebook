function H = Butterworth( D0, height, width )
    for i = 1 : height
        x = i - (height / 2);
        for j = 1 : width
            y = j - (width / 2);
            H(i, j) = 1 / (1 + (D0 ^ 2) / (x ^ 2 + y ^ 2));
        end
    end
end
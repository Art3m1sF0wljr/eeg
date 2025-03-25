function largest_k_elements = largest_k(array, k)
    % Sort the array in descending order
    sorted_array = quicksort(array);
   
    % Return the first k elements (largest k)
    largest_k_elements = sorted_array(end-k+1:end);
end

function sortedArray = quicksort(array)
    % Base case: arrays with 0 or 1 element are already sorted
    if length(array) <= 1
        sortedArray = array;
        return;
    end
    
    % Choose pivot (middle element)
    pivotIndex = floor(length(array)/2);
    pivot = array(pivotIndex);
    
    % Partition the array
    less = [];
    equal = [];
    greater = [];
    
    for i = 1:length(array)
        if array(i) < pivot
            less = [less, array(i)];
        elseif array(i) == pivot
            equal = [equal, array(i)];
        else
            greater = [greater, array(i)];
        end
    end
    
    % Recursively sort and combine
    sortedArray = [quicksort(less), equal, quicksort(greater)];
end
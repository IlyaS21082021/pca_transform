#include <iostream>
#include <vector>
#include <cstdlib>
#include <random>

#include "pca_transform.h"

int main()
{    
    std::vector<float> dataset(20);
    dataset[0] = 523.811; dataset[1] = 261.462; dataset[2] = 157.097; dataset[3] = 689.927;
    dataset[4] = 831.986; dataset[5] = 330.817; dataset[6] = 706.568; dataset[7] = 350.39;
    dataset[8] = 628.032; dataset[9] = 7.57863; dataset[10] = 293.052; dataset[11] = 514.535;
    dataset[12] = 932.176; dataset[13] = 907.646; dataset[14] = 211.179; dataset[15] = 393.164;
    dataset[16] = 247.236; dataset[17] = 319.959; dataset[18] = 747.366; dataset[19] = 569.854;

    PCA_Transform pca(dataset, 5, 4, 3);
    auto res = pca.ApplyPCATransform(dataset, 5, 4);

    for (auto v : res)
        std::cout << v << " ";
    std::cout << "\n";

    return 0;
}

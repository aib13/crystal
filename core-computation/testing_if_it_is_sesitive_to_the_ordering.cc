#include "utils.h"

void compute_indexes() {

	int[] freq = {0,0,0,0};

	for(int i = 0 ; i < 4; ++i) {

		for(int j = 0 ; j < 4; ++j) {

			for(int k = 0; k < 4; ++k) {

				for(int l = 0; l < 4; ++l) {

					freq[i] = 1;
					freq[j] = 1;
					freq[k] = 1;
					freq[l] = 1;

					int ok = 1;
					for( index = 0; index < 4; ++index) {

						if(freq[index] != 1) {
							ok = 0;
						}

					}

					if(ok == 1) {

						// do something

					}

				}

			}

		}

	}

}
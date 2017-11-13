#include <vector>
#include<algorithm>
#include <time.h>
#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void Compute_distance_data_gpu(int nthreads, const int K, const Dtype* bottom,
	      const Dtype* label, const Dtype* center, Dtype* distance) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / K;
    int k = index % K;
    const int label_value = static_cast<int>(label[m]);
    // distance(i) = x(i) - c_{y(i)}
    distance[index] = bottom[index] - center[label_value * K + k];
  }
}

template <typename Dtype>
__global__ void Compute_center_diff_gpu(int nthreads, const int M, const int K, 
        const Dtype* label, const Dtype* distance, Dtype* variation_sum, 
        Dtype* center_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int count = 0;
    for (int m = 0; m < M; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (label_value == index) {
        count++;
        for (int k = 0; k < K; k++) {
          variation_sum[index * K + k] -= distance[m * K + k];
        }
      }
    }
    if (count > 0) {
        for (int k = 0; k < K; k++) {
        center_diff[index * K + k] += variation_sum[index * K + k] / Dtype(count);
        }
     }
  }
}

template <typename Dtype>
__global__ void Compute_triplet_distance_data_gpu(int nthreads, const int K, const Dtype* bottom,
	      const Dtype* center, Dtype* distance) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index % K;
    // distance(i) = x(i) - c_{y(i)}
    distance[index] = bottom[m] - center[index];
  }
}

template <typename Dtype>
__global__ void Compute_direct_distance_data_gpu(int nthreads, const Dtype* bottom,
	      const Dtype* center, Dtype* distance) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // distance(i) = x(i) - c_{y(i)}
    distance[index] = bottom[index] - center[index];
  }
}

template <typename Dtype>
__global__ void Compute_triplet_center_diff_gpu(int nthreads, const int M, const int K, 
        const Dtype* label, const Dtype* distance, Dtype* triplet_variation_sum, 
        Dtype* center_diff, const Dtype* min_distance, const Dtype* min_center, Dtype * min_triplet_variation_sum) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int triplet_count = 0;
    int min_triplet_count = 0;
    for (int m = 0; m < M; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (label_value == index && min_center[m] >=0 ) {
        triplet_count++;
        for (int k = 0; k < K; k++) {
          triplet_variation_sum[index * K + k] -= distance[m * K + k];
        }
      }

      if (min_center[m] == index) {
         min_triplet_count++;

        for (int k = 0; k < K; k++) {
          min_triplet_variation_sum[index * K + k] += min_distance[m * K + k];
        }
        
      }
    }


    for (int k = 0; k < K; k++) {
      if (triplet_count > 0) 
        center_diff[index * K + k] += triplet_variation_sum[index * K + k] / Dtype(triplet_count);
      if (min_triplet_count > 0)
        center_diff[index * K + k] += min_triplet_variation_sum[index * K + k] / Dtype(min_triplet_count); 
    }
  }
}


template <typename Dtype>
    __global__ void Compute_center_norm_gpu(int nthreads, const int K_, const Dtype * center, Dtype * center_mutable, const Dtype* center_norm) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int cur_norm_index = index / K_;
        center_mutable[index] = center[index] / center_norm[cur_norm_index]; 
  }
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  srand((unsigned)time(NULL));

  int nthreads = M_ * K_;

  // normalization

 // Compute_center_sqrt_gpu<Dtype><<<CAFFE_GET_BLOCKS(N_ * K_),
      //CAFFE_CUDA_NUM_THREADS>>>(N_ * K_, K_, this->blobs_[0]->gpu_data(), center_sqrt.mutable_gpu_data());
  


    if (this->center_norm_update == false && this->normalization == true) {
        Dtype average_norm = 0;
        if (this->cur_iter_times % 50 == 0) {
            caffe_gpu_powx<Dtype>(N_ * K_, this->blobs_[0]->gpu_data(), Dtype(2), center_sqrt.mutable_gpu_data());

            for (int i = 0; i < N_; i++) {
 //   caffe_gpu_dot(K_, this->blobs_[0]->gpu_data() + i * K_, this->blobs_[0]->gpu_data() + i * K_, &norm);
                
                caffe_gpu_asum<Dtype>(K_, this->center_sqrt.gpu_data() + i * K_, this->center_norm.mutable_cpu_data() + i);
                    this->center_norm.mutable_cpu_data()[i] = pow(this->center_norm.cpu_data()[i], Dtype(0.5));

                average_norm += this->center_norm.cpu_data()[i];


//    caffe_gpu_dot(K_, this->blobs_[0]->gpu_data() + i * K_, this->blobs_[0]->gpu_data() + i * K_, &norm);

            }
            //caffe_gpu_scale<Dtype>(K_, 1.0 / norm, this->blobs_[0]->gpu_data() + i * K_, this->blobs_[0]->mutable_cpu_data() + i * K_);

            Compute_center_norm_gpu<Dtype><<<CAFFE_GET_BLOCKS(N_ * K_),
                CAFFE_CUDA_NUM_THREADS>>>(N_ * K_, K_, this->blobs_[0]->gpu_data(), this->blobs_[0]->mutable_gpu_data(), this->center_norm.gpu_data());

            average_norm /= N_;
            std::cout << this->cur_iter_times << " " << average_norm << std::endl;
            this->center_norm_update = true;
        }
   }

  Compute_distance_data_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
                                this->blobs_[0]->gpu_data(), distance_.mutable_gpu_data());
  Dtype loss = 0;
  Dtype triplet_loss = 0;
  //std::cout << "before test" << std::endl;
  for (int i = 0; i < M_; i++) {
    caffe_gpu_dot(K_, distance_.gpu_data() + i * K_, distance_.gpu_data() + i * K_, distance_sq.mutable_cpu_data() + i);
    loss += distance_sq.cpu_data()[i];
  }

  loss = loss / M_ / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

    if (with_triplet) {
        nthreads = N_ * K_;

        caffe_set(M_ , (Dtype)-1., min_center.mutable_cpu_data());
       
       for (int i = 0; i < M_; i++) {
            
            Compute_triplet_distance_data_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
                CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, bottom[0]->gpu_data() + K_ * i,
                                this->blobs_[0]->gpu_data(), triplet_distance.mutable_gpu_data());

           min_distance_sq.mutable_cpu_data()[i] = 10000;

           //std::cout << "test" << std::endl;
           for (int t = 0; t < N_ / divide; t++) {

                int j = rand() % N_;
                
                Dtype triplet_distance_value;
                caffe_gpu_dot(K_, triplet_distance.gpu_data() + j * K_, triplet_distance.gpu_data() + j * K_, &triplet_distance_value);
                
                if (std::max(distance_sq.cpu_data()[i] + alpha -  triplet_distance_value, Dtype(0.)) > 0 && triplet_distance_value > distance_sq.cpu_data()[i]) {

                    if (min_distance_sq.cpu_data()[i] > triplet_distance_value) {

                        min_distance_sq.mutable_cpu_data()[i] = triplet_distance_value;
                        min_center.mutable_cpu_data()[i] = j;
                        caffe_copy(K_, triplet_distance.cpu_data() + j * K_,  min_distance.mutable_cpu_data() + i * K_);
                        break;

                    }
                }
               // std::cout << t << std::endl;
           }
            
           triplet_loss += std::max(distance_sq.cpu_data()[i] + alpha - min_distance_sq.cpu_data()[i], Dtype(0.));
        
       }

      top[1]->mutable_cpu_data()[0] = triplet_loss / M_ / Dtype(2);

    }

    //std::cout  << top[1]->cpu_data()[0] << std::endl;

}



template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int nthreads = N_;
  caffe_gpu_set(N_ * K_, (Dtype)0., variation_sum_.mutable_cpu_data());

  if (this->normalization == true) { 
     this->center_norm_update = false;
    this->cur_iter_times += 1;
  }

  Compute_center_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, bottom[1]->gpu_data(), distance_.gpu_data(), 
                                variation_sum_.mutable_cpu_data(), this->blobs_[0]->mutable_gpu_diff());

    if (with_triplet) {
    caffe_gpu_set(N_ * K_, (Dtype)0., triplet_variation_sum_.mutable_cpu_data());
    caffe_gpu_set(N_ * K_, (Dtype)0., min_triplet_variation_sum_.mutable_cpu_data());
    Compute_triplet_center_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, bottom[1]->gpu_data(), distance_.gpu_data(), 
                                triplet_variation_sum_.mutable_cpu_data(), this->blobs_[0]->mutable_gpu_diff(), 
                                min_distance.gpu_data(), min_center.gpu_data(), min_triplet_variation_sum_.mutable_cpu_data());
    }

  if (propagate_down[0]) {
        caffe_gpu_scale(M_ * K_, top[0]->cpu_diff()[0] / M_, 
                             distance_.gpu_data(), bottom[0]->mutable_cpu_diff());
      if (with_triplet) {
        caffe_gpu_sub(M_ * K_, distance_.gpu_data(), min_distance.gpu_data(), distance_.mutable_cpu_data());
       // std::cout << "test" << std::endl;
        for (int i = 0; i < M_; i++) {
            
            if (min_center.cpu_data()[i] >= 0) {
                caffe_gpu_axpy(K_, top[1]->cpu_diff()[0] / M_, distance_.cpu_data() + i * K_, bottom[0]->mutable_cpu_diff() + i * K_);
            
            }
        }
     }
    }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CenterLossLayer);

}  // namespace caffe

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CenterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.center_loss_param().num_output();  
  with_triplet =  this->layer_param_.center_loss_param().with_triplet();  
  alpha =  this->layer_param_.center_loss_param().alpha();  
  divide = this->layer_param_.center_loss_param().divide();  
  normalization = this->layer_param_.center_loss_param().normalization();  

  
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.center_loss_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    vector<int> center_shape(2);
    center_shape[0] = N_;
    center_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(center_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(
        this->layer_param_.center_loss_param().center_filler()));
    center_filler->Fill(this->blobs_[0].get());

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  this->center_norm_update = false;
  this->cur_iter_times = 0;
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  M_ = bottom[0]->num();
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  LossLayer<Dtype>::Reshape(bottom, top);
  distance_.ReshapeLike(*bottom[0]);
  variation_sum_.ReshapeLike(*this->blobs_[0]);
  center_sqrt.ReshapeLike(*this->blobs_[0]);



  if (with_triplet) {

    vector<int> shape(0);
    top[1]->Reshape(shape);

    shape.push_back(M_);
    min_center.Reshape(shape);
    min_distance.ReshapeLike(*bottom[0]);
    min_distance_sq.Reshape(shape);
    distance_sq.Reshape(shape);
    shape[0] = K_;
    tmp_distance.Reshape(shape);
    triplet_variation_sum_.ReshapeLike(*this->blobs_[0]); 
    min_triplet_variation_sum_.ReshapeLike(*this->blobs_[0]); 

    triplet_distance.ReshapeLike(*this->blobs_[0]);

    shape[0] = N_;
    center_norm.Reshape(shape);

  }

  else{

    vector<int> shape(1);
    shape[0] = M_;
    distance_sq.Reshape(shape);
    
    shape[0] = N_;
    center_norm.Reshape(shape);
  }

}

template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* center = this->blobs_[0]->cpu_data();
  Dtype* distance_data = distance_.mutable_cpu_data();
  
  // the i-th distance_data
  if (with_triplet) {
    caffe_set(M_ , (Dtype)-1., min_center.mutable_cpu_data());
  }
  Dtype alpha = 0.2;
  Dtype triplet_loss = 0;
  Dtype center_loss = 0;
  for (int i = 0; i < M_; i++) {
    const int label_value = static_cast<int>(label[i]);
    // D(i,:) = X(i,:) - C(y(i),:)
    caffe_sub(K_, bottom_data + i * K_, center + label_value * K_, distance_data + i * K_);

    distance_sq.mutable_cpu_data()[i] = caffe_cpu_dot(K_, distance_data + i * K_, distance_data + i * K_);
    
    if (with_triplet) { 
        min_distance_sq.mutable_cpu_data()[i] = 1000;
        for (int k = 0; k < N_; k++) {
        
            if (k == label_value)
                continue;
            caffe_sub(K_, bottom_data + i * K_, center + K_ * k, tmp_distance.mutable_cpu_data());
        
            Dtype tmp = caffe_cpu_dot(K_, tmp_distance.cpu_data(), tmp_distance.cpu_data());

            if (std::max(distance_sq.cpu_data()[i] + alpha - tmp, Dtype(0.)) > 0 && tmp > distance_sq.cpu_data()[i]) {

                if (min_distance_sq.cpu_data()[i] > tmp) {

                    min_distance_sq.mutable_cpu_data()[i] = tmp;
                    min_center.mutable_cpu_data()[i] = k;
                    caffe_copy(K_, tmp_distance.cpu_data(),  min_distance.mutable_cpu_data() + i * K_);

                }
            }
        }
        triplet_loss += std::max(distance_sq.cpu_data()[i] + alpha - min_distance_sq.cpu_data()[i], Dtype(0.));
    }
    center_loss += distance_sq.cpu_data()[i]; 
        
  }

  //Dtype dot = caffe_cpu_dot(M_ * K_, distance_.cpu_data(), distance_.cpu_data());
  //Dtype loss = dot / M_ / Dtype(2);
  top[0]->mutable_cpu_data()[0] = center_loss / M_ / Dtype(2);
  if (with_triplet) {
    top[1]->mutable_cpu_data()[0] = triplet_loss / M_ / Dtype(2);
  }

}

template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Gradient with respect to centers
  if (this->param_propagate_down_[0]) {
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* variation_sum_data = variation_sum_.mutable_cpu_data();
    const Dtype* distance_data = distance_.cpu_data();

    // \sum_{y_i==j}
    for (int n = 0; n < N_; n++) {
      int count = 0;
      caffe_set(K_, (Dtype)0., variation_sum_.mutable_cpu_data());
      for (int m = 0; m < M_; m++) {
        const int label_value = static_cast<int>(label[m]);
        if (label_value == n) {
          count++;
          caffe_sub(K_, variation_sum_data, distance_data + m * K_, variation_sum_data);
        }
      }
      caffe_axpy(K_, (Dtype)1./(count + (Dtype)1.), variation_sum_data, center_diff + n * K_);

      if (with_triplet) {
        int  triplet_count = 0;
        
        caffe_set(K_, (Dtype)0., variation_sum_.mutable_cpu_data());
        caffe_set(K_, (Dtype)0., triplet_variation_sum_.mutable_cpu_data());

        for (int m = 0; m < M_; m++) {
            if (min_center.cpu_data()[m] == n) {
                triplet_count++;

               // caffe_sub(K_, variation_sum_data, distance_data + m * K_, variation_sum_data);
                caffe_add(K_, triplet_variation_sum_.cpu_data(), min_distance.cpu_data() + m * K_, triplet_variation_sum_.mutable_cpu_data());

            }
            if (label[m] == n) {
                triplet_count++;
               caffe_sub(K_, variation_sum_data, distance_data + m * K_, variation_sum_data);
                
            }
        }

        caffe_axpy(K_, (Dtype)1./(triplet_count + (Dtype)1.), triplet_variation_sum_.cpu_data(), center_diff + n * K_);
        //caffe_add(k_, tmp_distance.cpu_data(), center_diff + n * K_, center_diff + n * K_);

        caffe_axpy(K_, (Dtype)1./(triplet_count + (Dtype)1.), variation_sum_data, center_diff + n * K_);
        //caffe_add(k_, tmp_distance.cpu_data(), center_diff + n * K_, center_diff + n * K_);
        
      }
    }
  }
  // Gradient with respect to bottom data 
  if (propagate_down[0]) {
    caffe_copy(M_ * K_, distance_.cpu_data(), bottom[0]->mutable_cpu_diff());
    //caffe_axpy(M_ * K_, Dtype(1.0), distance_.cpu_data(), bottom[0]->mutable_cpu_diff());
    caffe_scal(M_ * K_, top[0]->cpu_diff()[0] / M_, bottom[0]->mutable_cpu_diff());

    if (with_triplet) { 
        for (int i = 0; i < M_; i++) {
            int k = min_center.cpu_data()[i];
            if (k >= 0) {

                caffe_sub(K_, distance_.cpu_data() + i * K_, min_distance.cpu_data() + i * K_, tmp_distance.mutable_cpu_data());
                //caffe_scale(K_, top[1]->cpu_diff()[0] / M_, tmp_distance.mutable_cpu_data());
                //caffe_add(K_, tmp_distance.cpu_data(), bottom[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
                caffe_axpy(K_, top[1]->cpu_diff()[0] / M_, tmp_distance.cpu_data(), bottom[0]->mutable_cpu_diff() + i * K_);

            }
        }
     }
  }

  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(CenterLossLayer);
#endif

INSTANTIATE_CLASS(CenterLossLayer);
REGISTER_LAYER_CLASS(CenterLoss);

}  // namespace caffe

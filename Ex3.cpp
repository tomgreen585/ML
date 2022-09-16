//#pragma GCC optimize("Ofast")

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <math.h>
#include <numeric>
#include <fstream> 
#include <sstream>
#include <chrono>
#include <random>

//debugging functions
void stop(std::string message){
	std::cout<<message<<" enter int to continue"<<std::endl;
	int d;
	std::cin>>d;
}

void save_vector_to_file(std::vector<double> v){
	std::string file_name;
	std::cout<<" Enter file name to save the vector: ";
	std::cin>>file_name;
	std::ofstream ofs;
    std::cout<<"Saving vector "<<file_name<<" ..."<<std::endl;
    ofs.open(file_name.c_str(), std::ofstream::out);//, std::ofstream::out | std::ofstream::trunc);
	if (ofs.is_open()){
		int count =0 ;
		for(double ve:v){
			ofs<<count<<" "<<ve<<std::endl;
			count++;
		}
		ofs.close();
	}
}


double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

double d_sigmoid(double x){
    return (1.0-sigmoid(x))*sigmoid(x);
}


struct train_point_t{
	std::vector<double> inputs;
	std::vector<double> target;
};

struct train_set_t{
  std::vector<train_point_t> train_points;	
};


void set_train_set(train_set_t& ts){
	train_point_t p1;
	p1.inputs = {0.0, 0.0}; //initialize vector
	p1.target = {1.0};
	ts.train_points.push_back(p1);
	train_point_t p2;
	p2.inputs = {0.0, 1.0};
	p2.target = {0.0};
	ts.train_points.push_back(p2);
	train_point_t p3;
	p3.inputs = {1.0, 0.0};
	p3.target = {0.0};
	ts.train_points.push_back(p3);
	train_point_t p4;
	p4.inputs = {1.0, 1.0};
	p4.target = {1.0};
	ts.train_points.push_back(p4);
	
}


void print_train_point(train_point_t tp){
	std::cout<<"Training point. Inputs: ";
	for (double inp:tp.inputs) { std::cout<<inp<<" ";};
	std::cout<<" Targets: ";
	for (double out:tp.target) { std::cout<<out<<" ";};
	std::cout<<std::endl;
}


struct Neuron{
	std::vector<double> weights;   // weights
	std::vector<double> d_weights; // weight derivatives
	double bias;
	double d_bias; // bias derivative
	double z; // before sigmoid
	double y; // current output
	double delta;
    double calculate(const std::vector<double>& inputs);
    void set_neuron(std::vector<double> wi, double bi);
	double error(double t){return (y-t);};
	void print();
};

void Neuron::set_neuron(std::vector<double> wi, double bi){
	weights = wi; //std::transform(wi.begin,wi.end(),w.begin(),[](double in){return out; });  
	bias = bi;
}

double Neuron::calculate(const std::vector<double>& inputs){
	z = std::inner_product(weights.begin(), weights.end(),inputs.begin(),bias);
	y =  sigmoid(z);
	return y;
}

void Neuron::print(){
	std::cout<<"Neuron:\tdelta="<<delta<<std::endl;
	std::cout<<"       \tbias="<<bias<<" d_bias="<<d_bias<<std::endl;
	std::cout<<"       \tWeights: ";
	std::for_each(weights.begin(),weights.end(),[](double w){std::cout<<w<<" ";});
	std::cout<<std::endl;
	std::cout<<"        \tDerivatives: ";
	std::for_each(d_weights.begin(),d_weights.end(),[](double dw){std::cout<<dw<<" ";});
	std::cout<<std::endl;
	std::cout<<"        \tz="<<z<<" y="<<y<<std::endl;
}


struct Net{
	std::vector<Neuron> hidden_layer;
	std::vector<Neuron> output_layer;
	std::vector<double> hidden_outputs;
	std::vector<double> output_errors;
	// net metaparameters
	double in_median;
	double in_deviation;
	double learning_rate;
//	int n_batch;
	int n_hidden;
	int n_epoch;  // maximum number of training epochs
	
	double train_set_error;
    std::vector<double> convergence;
	void init(uint n_inputs, uint n_hidden,uint n_outputs);
    void print();
    void forward_prop(const std::vector<double>& inputs);
    double train_point_error(const train_point_t& tp);
    double calc_train_set_error(const train_set_t& ts);
    void backprop(const train_point_t& tp);
    void step_by_gradient(double lr);
    void train(train_set_t ts);
    void print_outputs();
};


// reserve memory
// put random values into vectors for weights and biases
void Net::init( uint n_inputs, uint n_hidden,uint n_outputs){
	
	std::cout<<" Initializing: n_inputs="<<n_inputs<<" n_hidden="<<n_hidden;
	std::cout<<" n_outputs="<<n_outputs<<std::endl;
	//srand (time(NULL));
	std::default_random_engine generator;
    std::normal_distribution<double> distribution(in_median,in_deviation);
	
	hidden_layer.reserve(n_hidden);
	for (uint i = 0 ; i < n_hidden ; i++){
	   Neuron n1;
	   n1.bias = distribution(generator);
	   n1.d_bias = 0.0;
	   n1.delta = 0.0;
	   n1.weights.reserve(n_inputs);
	   n1.d_weights.reserve(n_inputs);
	   for (uint j = 0; j < n_inputs ; j++){
		   n1.weights.emplace_back(distribution(generator));
		   n1.d_weights.emplace_back(0.0);
	   }
	   n1.y = n1.z = 0.0;
	   hidden_layer.emplace_back(n1);
    }
    
    output_layer.reserve(n_outputs);
    output_errors.reserve(n_outputs);
	for (uint i = 0 ; i < n_outputs ; i++){
	   Neuron n;
	   n.bias = distribution(generator);
	   n.d_bias = 0.0;
	   n.delta =0.0;
	   n.weights.reserve(n_hidden);
	   n.d_weights.reserve(n_hidden);
	   for (uint j = 0; j < n_hidden ; j++){
//		   n.weights.emplace_back(distribution(generator));
		   n.weights.emplace_back(0.0);
		   n.d_weights.emplace_back(0.0);
	   }
	   n.y = n.z = 0.0;
	   output_layer.emplace_back(n);
	   output_errors.emplace_back( 0.0);
    }
}

void Net::print(){
	std::cout<<"***HIDDEN***"<<std::endl;
	for (uint i = 0 ; i < hidden_layer.size() ; i++){
	   hidden_layer.at(i).print();
    }
    std::cout<<"****OUTPUT******"<<std::endl;
	for (uint i = 0 ; i < output_layer.size() ; i++){
	   output_layer.at(i).print();
    }

}

void Net::forward_prop(const std::vector<double>& inputs){
	
	hidden_outputs.clear();
	for(Neuron& nhl:hidden_layer) { //each neuron in hidden layer (nhl)
		hidden_outputs.emplace_back(nhl.calculate(inputs));
	}
	for(Neuron& nol:output_layer) {
		nol.calculate(hidden_outputs);
    }
}


// 
double Net::train_point_error(const train_point_t& tp){
   double error = 0.0;
   forward_prop(tp.inputs);
   // for all neutons in output layer
   for (uint i=0; i< output_layer.size() ; i++){
		double err = output_layer.at(i).y-tp.target[i];
		error = error + err*err;
	}
	return error;
}

double Net::calc_train_set_error(const train_set_t& ts){
	double tse = 0.0;
	for (uint i = 0 ; i < ts.train_points.size() ; i++){
		tse = tse + train_point_error(ts.train_points.at(i));
	}
	return tse/ ts.train_points.size();
}


void Net::print_outputs(){
	std::cout<<" Outputs of the Network: ";
	double max = 0.0;
	int index_max = -1;
    for (uint i=0; i< output_layer.size() ; i++){
       std::cout<<output_layer.at(i).y<<" ";
       if (output_layer.at(i).y>max){ 
		   max = output_layer.at(i).y;
		   index_max = i;
	   }
    }
    std::cout<<" Index of the max. output: "<<index_max<<std::endl; 
    std::cout<<std::endl;    
}


//v1 = v1+sca*v2
void modify_vector(std::vector<double>& v1, const std::vector<double>& v2, double sca){
	
	for (uint i = 0 ; i < v1.size(); i++){
	   v1[i] = v1[i] + sca*v2[i];
	}
}


void Net::backprop(const train_point_t& tp){
	forward_prop(tp.inputs);
	// for all neurons in output layer - calculate delta
	int count =0;
	for(Neuron& nol:output_layer){
		nol.delta = d_sigmoid(nol.z)*(output_layer.at(count).y-tp.target[count]);
	}
	
	//for all neurons in hidden layer - calculate delta
	int count_hidden = 0;
	for (Neuron& nhl:hidden_layer){
		// sum of weighted deltas from output layer
		nhl.delta = 0.0;
		//int count1 = 0;
		for(uint i = 0 ; i <  output_layer.size() ; i++){
			nhl.delta = nhl.delta + output_layer.at(i).weights.at(count_hidden) * output_layer.at(i).delta;
		}
		nhl.delta = nhl.delta*d_sigmoid(nhl.z);
		count_hidden++;
	}
	
	// derivaties of output layer
	for(Neuron& nol:output_layer){
		nol.d_bias = 2.0*nol.delta;
		int count =0;
		for(Neuron& nhl:hidden_layer){
			nol.d_weights.at(count) = 2.0*nol.delta*nhl.y;
			count++;
		}
    }
     
    // deivatives of hidden layer
	for(Neuron& nhl:hidden_layer){
		nhl.d_bias = 2.0 * nhl.delta;
		for ( uint i = 0 ; i < nhl.weights.size(); i++){
			    nhl.d_weights.at(i) = 2.0*nhl.delta*tp.inputs.at(i);
		}
	}
}


void Net::step_by_gradient(double lr){
	for(Neuron& nhl:hidden_layer){
		nhl.bias = nhl.bias - lr * nhl.d_bias;
		modify_vector(nhl.weights,nhl.d_weights,-lr);
    }
	for(Neuron& nol:output_layer){
		nol.bias = nol.bias - lr * nol.d_bias;
		modify_vector(nol.weights,nol.d_weights,-lr);
    }
}

void Net::train(train_set_t ts){
	std::cout<<"Starting training..."<<std::endl;
	int epoch = 0;
	double dataset_error = calc_train_set_error(ts);
	std::cout<<" Before training error="<<dataset_error<<std::endl;
	convergence.push_back(dataset_error);
	
	while ((dataset_error>0.001)&&(epoch<n_epoch)){
		for (uint i = 0; i < ts.train_points.size() ; i++){
		    //calculate_all_gradients( ts.train_points.at(i) );
		    backprop( ts.train_points.at(i) );
		    step_by_gradient(learning_rate);
	    } 
	    // for display only - training can run without
	    dataset_error = calc_train_set_error(ts);
	    //std::cout<<" epoch="<<epoch<<"/"<<n_epoch<<" error="<<dataset_error<<std::endl;
	    convergence.push_back(dataset_error);
	    epoch++;
    } //epoch
    for ( uint i = 0 ; i < ts.train_points.size() ; i++){
	    // forward_prop(ts.train_points.at(i).inputs);
        // double e = ts.train_points.at(i).target[0] - output_layer[0].y;
         //std::cout<<" i= "<<i<<" target="<<ts.train_points.at(i).target[0]<<" y="<<output_layer[0].y<<" e="<<e<<std::endl;
    }
    //print_neuron();

    save_vector_to_file(convergence);
}

void draw_output_surface(Net& n){
	// save 
	std::ofstream of;
	of.open("outs.txt", std::ofstream::out);
	if (of.is_open()){
	  double dx = 0.01;
	  // save surface into the file
 	  for (double x0 = 0.0; x0 < 1.0; x0 = x0 + dx)
	   for (double x1 = 0.0; x1 < 1.0; x1 = x1 + dx){
		   n.forward_prop({x0,x1});
		   of<<x0<<" "<<x1<<" "<<n.output_layer[0].y<<std::endl;
	   }
	  of.close(); 
	  system("gnuplot gplot");
    }
}

int main(){
	Net net;
	train_set_t ts;
	set_train_set(ts);
	
	net.n_epoch = 1500;
	net.learning_rate = 3.1;
	net.in_median = 0.0;
	net.in_deviation = 4.0;
	
	net.init(ts.train_points[0].inputs.size(),2,ts.train_points[0].target.size());
	std::cout<<" lr="<<net.learning_rate;
	std::cout<<" n_epoch="<<net.n_epoch;
	std::cout<<" n_hid="<<net.n_hidden;
	std::cout<<" in_median = "<<net.in_median;
	std::cout<<" in_deviation = "<<net.in_deviation<<std::endl;
	//print network
	double dataset_error = net.calc_train_set_error(ts);
	std::cout<<" Before training error="<<dataset_error<<std::endl;
	net.forward_prop(ts.train_points[0].inputs);
	for(Neuron n:net.hidden_layer) n.print();
	for(Neuron n:net.output_layer) n.print();
	
	net.train(ts);
	std::cout<<" After training:"<<std::endl;
	dataset_error = net.calc_train_set_error(ts);
	std::cout<<" After training error="<<dataset_error<<std::endl;
	std::cout<<" Hidden layer**************"<<std::endl;
	for(Neuron n:net.hidden_layer) n.print();
	std::cout<<" Output layer**************"<<std::endl;
	for(Neuron n:net.output_layer) n.print();
	//draw_output_surface(net);
	//stop("after training state");

}

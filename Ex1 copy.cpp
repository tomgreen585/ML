// single neuron code with backprpagation
// Ex1 of Assignment 2 implemented

#include <iostream>
#include <fstream> 
#include <math.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <vector>

double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

struct train_point_t{
	std::vector<double> inputs;
	double target;
};

struct train_set_t{
  std::vector<train_point_t> train_points;	
};

void set_train_set(train_set_t& ts){
	train_point_t p1;
	p1.inputs = {0.0, 0.0};
	p1.target = 1.0;
	ts.train_points.push_back(p1);
	train_point_t p2;
	p2.inputs = {0.0, 1.0};
	p2.target = 0.0;
	ts.train_points.push_back(p2);
	train_point_t p3;
	p3.inputs = {1.0, 0.0};
	p3.target = 0.0;
	ts.train_points.push_back(p3);
	train_point_t p4;
	p4.inputs = {1.0, 1.0};
	p4.target = 1.0;
	ts.train_points.push_back(p4);
	
}

void print_set(const train_set_t& ts){
	for(train_point_t tp:ts.train_points){
		std::cout<<" Inputs ";
		for(double in:tp.inputs) std::cout<<in<<" ";
		std::cout <<" Output: "<<tp.target<<" ";
		std::cout<<std::endl;
	}
}

// vector to store 
std::vector<double> convergence;
void save_vector_to_file(std::vector<double> v){
	std::string file_name;
	std::cout<<" Enter file name for convergence graph:";
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



struct Neuron{
	//int nw;
	std::vector<double> weights;  
	double bias;
	double z; 
	double y; 
	void init(int nw);
	double forward(std::vector<double> inputs);
	void print_neuron();
};

void draw_output(Neuron& n);


void Neuron::init(int n){
	bias = 0.0;
	weights.reserve(n);
	weights.emplace_back(0.0);
	weights.emplace_back(0.0);
}

double Neuron::forward(std::vector<double> inputs){
	z = bias;
	for (unsigned int i = 0 ; i < weights.size() ; i++){
		z = z + weights[i]*inputs[i];
	}
//	z = std::inner_product(weights, weights + nw,inputs.begin(),bias);
	y =  sigmoid(z);
	return y;
}


void Neuron::print_neuron(){
    std::cout<<" bias="<<bias;
    std::cout<<" w0="<<weights[0];
    std::cout<<" w1="<<weights[1]<<std::endl;
	
}

double error(Neuron& neuron, double t){
	return (neuron.y -t);
}

double total_error(Neuron& neuron, const train_set_t& train_set){
    double tot_error = 0.0;
    for ( unsigned int i =0 ; i < train_set.train_points.size() ; i++){
        neuron.forward(train_set.train_points.at(i).inputs);
        double e = train_set.train_points.at(i).target - neuron.y;
        tot_error = tot_error + e*e;
     }
     return tot_error;
}

void gradient_search(Neuron& neuron,const train_set_t& train_set){
    double db, dw0,dw1;
    double learn_rate = 5.0; 
    double current_tot_err = total_error(neuron,train_set);
    int n_step = 0;
    
    while ((current_tot_err>0.01)&&(n_step<150)){
		
		for (unsigned int i = 0 ; i < train_set.train_points.size(); i++){

			neuron.forward(train_set.train_points[i].inputs);

		    double e0 = error(neuron,train_set.train_points[i].target);

		    double delta = (sigmoid(neuron.z) * (1.0-sigmoid(neuron.z))*e0);

			db = delta*2.0;

			dw0 = delta*(train_set.train_points[i].inputs[0])*2.0;

			dw1 = delta*(train_set.train_points[i].inputs[1])*2.0;

			neuron.bias = neuron.bias - learn_rate*db;
			neuron.weights[0] = neuron.weights[0] - learn_rate*dw0;
			neuron.weights[1] = neuron.weights[1] - learn_rate*dw1;

			current_tot_err = total_error(neuron,train_set);
			std::cout<<" n_step="<<n_step<<"  error="<<current_tot_err;
			std::cout<<" w0="<<neuron.weights[0];
			std::cout<<" w1="<<neuron.weights[1];
			std::cout<<" bias="<<neuron.bias<<std::endl;
			convergence.push_back(current_tot_err);
			n_step++;                   
		}
    }
    
    for ( unsigned int i = 0 ; i < train_set.train_points.size() ; i++){
	     neuron.forward(train_set.train_points.at(i).inputs);
         double e = train_set.train_points.at(i).target - neuron.y;
         std::cout<<" i= "<<i<<" y="<<neuron.y<<" t="<<train_set.train_points.at(i).target<<" e="<<e<<std::endl;
    }
    neuron.print_neuron();

}

void draw_output(Neuron& n){
	std::ofstream of;
	of.open("outs.txt", std::ofstream::out);
	if (of.is_open()){
	  double dx = 0.01;
 	  for (double x0 = 0.0; x0 < 1.0; x0 = x0 + dx)
	   for (double x1 = 0.0; x1 < 1.0; x1 = x1 + dx){
		    of<<x0<<" "<<x1<<" "<<n.forward({x0,x1})<<std::endl;
	   }
	  of.close(); 
	  system("gnuplot gplot");
    }
}

  
int main(){
	train_set_t train_set;
	set_train_set(train_set);
	print_set(train_set);
	
	Neuron neuron;
	neuron.init(2);
   	gradient_search(neuron,train_set);
   	save_vector_to_file(convergence);
	 
 } 

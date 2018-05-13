/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	num_particles= 150;
	particles.resize(num_particles);
	normal_distribution <double> dist_x(x,std[0]);
	normal_distribution <double> dist_y(y,std[1]);
	normal_distribution <double> dist_theta(theta,std[2]);

	for(unsigned int i=0; i<num_particles; i++){
		Particle p;
		p.id=i;
		p.x=dist_x(gen);
		p.y=dist_y(gen);
		p.theta=dist_theta(gen);
		p.weight=1;
		particles.push_back(p);
	}
	is_initialized=true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	for(unsigned int i=0; i<num_particles; i++){
		// To avoid division by zero
		if(abs(yaw_rate)<0.00001){
			particles[i].x += cos(yaw_rate)*velocity*delta_t;
			particles[i].y += sin(yaw_rate)*velocity*delta_t;
	}
	else{
		particles[i].x += velocity/yaw_rate * (sin(particles[i].theta  + yaw_rate*delta_t) - sin(particles[i].theta));
		particles[i].y += velocity/yaw_rate * (-cos(particles[i].theta  + yaw_rate*delta_t) + cos(particles[i].theta));
		particles[i].theta += yaw_rate * delta_t;
	}
	normal_distribution <double> dist_x(particles[i].x,std_pos [0]);
	normal_distribution <double> dist_y(particles[i].y,std_pos[1]);
	normal_distribution <double> dist_theta(particles[i].theta, std_pos[2]);
	
	particles[i].x=dist_x(gen);
	particles[i].y=dist_y(gen);
	particles[i].theta=dist_theta(gen);
}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(unsigned int i=0; i<observations.size(); i++){
		LandmarkObs obs=observations[i];
		double distance_min=99999;
		int map_id;

		for(unsigned int i=0; i<predicted.size(); i++){
			LandmarkObs pred=predicted[i];

			double curr_distance= dist(obs.x, obs.y, pred.x, pred.y);
			if(curr_distance<distance_min){
				distance_min=curr_distance;
				map_id=pred.id;
			}
		}
		observations[i].id=map_id;	
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	//Transforming the vehicle corrdinates to the map coordinates
	for (unsigned int i=0; i<num_particles; i++){
		vector<LandmarkObs> trans_obs;

		vector<LandmarkObs> predictions;

		for (unsigned int j=0; j<observations.size(); j++){
			double trans_x= particles[i].x + cos(particles[i].theta)*observations[j].x - sin(particles[i].y)*observations[j].y;
			double trans_y= particles[i].y + sin(particles[i].theta)*observations[j].x + cos(particles[i].y)*observations[j].y;
			trans_obs.push_back(LandmarkObs{observations[j].id, trans_x, trans_y});
		}

		for(unsigned int j=0; j<map_landmarks.landmark_list.size(); j++){
			float map_x=map_landmarks.landmark_list[j].x_f;
			float map_y=map_landmarks.landmark_list[j].y_f;
			int	map_id = map_landmarks.landmark_list[j].id_i;
			if(fabs(map_x- particles[i].x)<=sensor_range && fabs(map_y-particles[i].y)<=sensor_range){
				predictions.push_back(LandmarkObs{map_id, map_x, map_y});
			}
		}	

	 	dataAssociation(predictions, trans_obs);
	 	particles[i].weight=1;

	 	for(unsigned int j=0; j<trans_obs.size(); j++){
			double trans_obs_x= trans_obs[j].x;
			double trans_obs_y= trans_obs[j].y;
			double pred_x;
			double pred_y;

			for (unsigned int k = 0; k < predictions.size(); k++) {
       			if (predictions[k].id == trans_obs[j].id) {
          			pred_x= predictions[k].x;
          			pred_x= predictions[k].y;
				}
      		}

    	double std_x = std_landmark[0];
      	double std_y = std_landmark[1];
      	double obs_w = ( 1/(2*M_PI*std_x*std_y)) * exp( -( pow(pred_x -trans_obs_x,2)/(2*pow(std_x, 2)) + (pow(pred_y-trans_obs_y,2)/(2*pow(std_y, 2))) ) );

      	particles[i].weight *= obs_w;
	 	}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<double> current_weights;
	vector<Particle> new_particles;

	for(unsigned int i=0; i<num_particles; i++){
		current_weights.push_back(particles[i].weight);
	}

	uniform_int_distribution<int> uniform_dist(0, num_particles-1);
	int index= uniform_dist(gen);

	double twice_max_weight=(*max_element(weights.begin(), weights.end()))*2;
	uniform_real_distribution<double> uniform_real(0.0,twice_max_weight);

	double beta=0;

	for(unsigned int i=0; i<num_particles; i++){
		beta += uniform_real(gen);
		while (beta > weights[index]) {
      		beta -= weights[index];
      		index = (index + 1) % num_particles;
    	}
    	new_particles.push_back(particles[index]);
  	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

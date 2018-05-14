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
// Global Random Engine generator
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 200; //How do I decide, how many to use? Not sure 
  //particles.resize(num_particles);- Creates a problem. Fixed it, as number of particles is different at each step
  
  // Normal distributions for sensor noise
  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);

  // init particles
  for (unsigned int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = x;
    p.y = y;
    p.theta = theta;
    p.weight = 1;

    // add noise
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);

    particles.push_back(p);
  }

  is_initialized = true;
  cout<<"Initialized";

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  // Normal distribution for sensor noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (unsigned int i = 0; i < num_particles; i++) {

    // To avoid Divsion by zero
    if (fabs(yaw_rate) < 0.00001) {  
      particles[i].x += cos(particles[i].theta)*velocity * delta_t;
      particles[i].y += sin(particles[i].theta)*velocity * delta_t;
    } 
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // Sensor Noise Addittion
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.

  for (unsigned int i = 0; i < observations.size(); i++) {
    
    //Current Observation
    LandmarkObs obs = observations[i];
    double minimum_dist = 100000;

    int map_id;
    
    for (unsigned int j = 0; j < predicted.size(); j++) {
      LandmarkObs pred = predicted[j];
      
      //Distance between Predicted landmark and map landmark
      double distance = dist(obs.x, obs.y, pred.x, pred.y);

      // find the predicted landmark nearest the current observed landmark
      if (distance < minimum_dist) {
        minimum_dist =distance ;
        map_id = pred.id;
      }
    }

    // Storing the associated map_id
    observations[i].id = map_id;
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
  //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
  //   for the fact that the map's y-axis actually points downwards.)
  //   http://planning.cs.uiuc.edu/node99.html

  // for each particle...
  for (unsigned int i = 0; i < num_particles; i++) {


    double part_x = particles[i].x;
    double part_y = particles[i].y;
    double part_theta = particles[i].theta;

    // Predicted Landmark Vectors
    vector<LandmarkObs> predictions;

	// Map to World Coordinates
    vector<LandmarkObs> trans_obs;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double trans_x = cos(part_theta)*observations[j].x - sin(part_theta)*observations[j].y + part_x;
      double trans_y = sin(part_theta)*observations[j].x + cos(part_theta)*observations[j].y + part_y;
      trans_obs.push_back(LandmarkObs{ observations[j].id, trans_x, trans_y });
    }

    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      //Id and x,y coordinates
      float map_x = map_landmarks.landmark_list[j].x_f;
      float map_y = map_landmarks.landmark_list[j].y_f;
      int map_id = map_landmarks.landmark_list[j].id_i;
      
      //Check if difference is less than 50(sensor range)?
      if (fabs(map_x - part_x) <= sensor_range && fabs(map_y - part_y) <= sensor_range) {

        
        predictions.push_back(LandmarkObs{ map_id, map_x, map_y });
      }
    }
    // dataAssociation for observations and predictions
    dataAssociation(predictions, trans_obs);

    //Initialise weight
    particles[i].weight = 1.0;

    for (unsigned int j = 0; j < trans_obs.size(); j++) {
      
      
      double trans_obs_x= trans_obs[j].x;
	  double trans_obs_y= trans_obs[j].y;
	  double pred_x;
	  double pred_y;

      int associated_id = trans_obs[j].id;

      //x,y coordinates of the prediction associated with the current observation
      for (unsigned int n = 0; n < predictions.size(); n++) {
        if (predictions[n].id == associated_id) {
          pred_x = predictions[n].x;
          pred_y = predictions[n].y;
        }
      }

      // Multivariate Gaussian
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double obs_w = (1/(2*M_PI*s_x*s_y)) * exp( -( pow(pred_x-trans_obs_x,2)/(2*pow(s_x, 2)) + (pow(pred_y-trans_obs_y,2)/(2*pow(s_y, 2)))));

      // Weight Calculation
      particles[i].weight *= obs_w;
    }
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
  vector<Particle> new_particles;


  vector<double> current_weights;
  for (unsigned i = 0; i < num_particles; i++) {
    current_weights.push_back(particles[i].weight);
  }

  // Index Selection
  uniform_int_distribution<int> uniform_dist(0, num_particles-1);
  int index = uniform_dist(gen);

  //Maximum Weight among Particles
  double max_weight = *max_element(current_weights.begin(), current_weights.end());

  
  uniform_real_distribution<double> uniform_real(0.0, max_weight);

  double beta = 0;

  // Resampling
  for (unsigned int i = 0; i < num_particles; i++) {
    beta += uniform_real(gen) * 2;
    while (beta > current_weights[index]) {
      beta -= current_weights[index];
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

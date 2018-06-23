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
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 500;

	normal_distribution<double> dist_x(x,std[0]);
	normal_distribution<double> dist_y(y,std[1]);
	normal_distribution<double> dist_theta(theta,std[2]);

	for(unsigned int i = 0;i<num_particles;i++){
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1;
		particles.push_back(particle);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	for(unsigned int i =0; i<num_particles;i++){

		double theta_old =particles[i].theta;
		double new_x, new_y, new_theta;

		if(fabs(yaw_rate)<0.000001){
			new_x = velocity*sin(particles[i].theta)*delta_t + particles[i].x;
			new_y = velocity*cos(particles[i].theta)*delta_t + particles[i].y;
			new_theta = theta_old;

		}else{
			new_theta = theta_old + (yaw_rate*delta_t);
			new_x = (velocity / yaw_rate) * (sin(new_theta) - sin(theta_old)) + particles[i].x;
			new_y = (velocity / yaw_rate) * (cos(theta_old) - cos(new_theta)) + particles[i].y;

		}

		normal_distribution<double> N_x(new_x,std_pos[0]);
		normal_distribution<double> N_y(new_y,std_pos[1]);
		normal_distribution<double> N_theta(new_theta,std_pos[2]);

		particles[i].x = N_x(gen);
		particles[i].y = N_y(gen);
		particles[i].theta = N_theta(gen);

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> landmarks, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(unsigned int i =0; i < observations.size();i++){

		double min_distance = numeric_limits<double>::max();

		for(unsigned int j = 0; j < landmarks.size(); j++){

			double distance = dist(landmarks[j].x,landmarks[j].y,observations[i].x,observations[i].y);

			if (distance<min_distance){
				min_distance = distance;
				observations[i].id = landmarks[j].id;
			}
		}

	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs>& observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double TotalWeight = 0;

	for(unsigned int i = 0; i < particles.size(); i++){

		std::vector<LandmarkObs> landmarks_inrange;

		double px = particles[i].x;
		double py = particles[i].y;
		double p_theta= particles[i].theta;

		for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++){

			LandmarkObs landmark;
			landmark.id = map_landmarks.landmark_list[j].id_i;
			landmark.x = (double)map_landmarks.landmark_list[j].x_f;
			landmark.y = (double)map_landmarks.landmark_list[j].y_f;

			double distance = dist(px,py,landmark.x,landmark.y);
			if (distance < sensor_range){
				landmarks_inrange.push_back(landmark);
			}
		}

		std::vector<LandmarkObs> observations_global;

		for(unsigned int j = 0; j < observations.size();++j){


			LandmarkObs observation;
			observation.id = observations[j].id;
			observation.x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + px;
			observation.y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + py;

			observations_global.push_back(observation);
		}

		dataAssociation(landmarks_inrange,observations_global);

		particles[i].weight = 1;

		std::vector<int> p_associations;
		std::vector<double> p_sense_x;
		std::vector<double> p_sense_y;

		for(unsigned int j = 0;j < observations_global.size(); j++){

			double obs_x,obs_y,land_x,land_y;

			obs_x = observations_global[j].x;
			obs_y = observations_global[j].y;

			p_associations.push_back(observations_global[j].id);
			p_sense_x.push_back(obs_x);
			p_sense_y.push_back(obs_y);

			for(unsigned int k = 0; k<landmarks_inrange.size();++k){
				if(landmarks_inrange[k].id == observations_global[j].id){
					land_x = landmarks_inrange[k].x;
					land_y = landmarks_inrange[k].y;
				}
			}
			double obs_prob; // non normalized weight

			obs_prob = multi_pdf(std_landmark[0],std_landmark[1],land_x,land_y,obs_x,obs_y);

			particles[i].weight *= obs_prob;
		}
		SetAssociations(particles[i], p_associations,p_sense_x,p_sense_y);
		p_associations.clear();
		p_sense_x.clear();
		p_sense_y.clear();

		TotalWeight += particles[i].weight;

	}

	//Normalizing weights
	for(unsigned int i = 0; i < particles.size();i++){
		particles[i].weight /= TotalWeight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::vector<double> weights;

	for(unsigned int i = 0;i < num_particles; i++){
		weights.push_back(particles[i].weight);
	}

	std::discrete_distribution<int> Resampler(weights.begin(),weights.end());

	std::vector<Particle> new_particles;

	for(unsigned int i = 0;i < num_particles; i++){

		int index = Resampler(gen);
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

    return particle;
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

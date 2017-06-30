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

random_device device;
mt19937 gen(device()); // TODO: MAYBE CAN USE THIS THROUGHOUT


// TODO: This seems like a GPU dream job... should try later on and see
// TODO: Make particle number scale in proportion to available resources
void ParticleFilter::init(double gps_x, double gps_y, double gps_theta, double std[]) {
	num_particles = 100; // 100 || 1000
	normal_distribution<double> dist_x(gps_x, std[0]);
	normal_distribution<double> dist_y(gps_y, std[1]);
	normal_distribution<double> dist_theta(gps_theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
		double sample_x, sample_y, sample_theta;
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);
		Particle new_particle {i, sample_x, sample_y, sample_theta, 1.0};
		particles.push_back(new_particle);
		weights.push_back(1.0);
	}
	is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	double v_x, v_y, v_theta;

	for (int i = 0; i < num_particles; i++) {
		v_x = dist_x(gen);
		v_y = dist_y(gen);
		v_theta = dist_theta(gen);

		if (fabs(yaw_rate) > 0.0001) {
			particles[i].x += (velocity / yaw_rate) *
							  (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
			particles[i].y += (velocity / yaw_rate) *
							  (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
			particles[i].theta += yaw_rate * delta_t;
		}
		else {
			particles[i].x += (velocity * delta_t * cos(particles[i].theta));
			particles[i].y += (velocity * delta_t * sin(particles[i].theta));
			particles[i].theta += 0.0001 * delta_t; // Thanks to rana for his slack question that showed me this fix
		}

		particles[i].x += v_x; // inject noise, important also to prevent duplicate particles (Thanks Udacity help vids!)
		particles[i].y += v_y;
		particles[i].theta += v_theta;
	}
}


double BivariateGaussian(double x_mean, double y_mean, double x_sample, double y_sample, double std_x, double std_y) {
	return (1. / (2. * M_PI * std_x * std_y)) *
		   exp(-(pow((x_sample - x_mean), 2.) / (2. * pow(std_x, 2.)) + (pow((y_sample - y_mean), 2.) / (2. * pow(std_y, 2.)))));
}
//            ex+ey = ex (1+ey-x) = ex + log(1+exp(y-x))

std::vector<LandmarkObs> FindDataAssociations(const std::vector<LandmarkObs> &predicted, const std::vector<LandmarkObs> &observations) {
	vector<LandmarkObs> associated_landmarks; // landmark associated with observation for particle.
	// TODO: begging for optimization via multiprocessing of some sort
	for (const auto &obs : observations) {
		LandmarkObs closest_landmark;

		double closest_distance = numeric_limits<double>::max();

		for (const auto &landmark : predicted) {
			double distance = dist(obs.x, obs.y, landmark.x, landmark.y);
			if (distance < closest_distance) {
				closest_landmark = landmark;
				closest_distance = distance;
			}
		}
		associated_landmarks.push_back(closest_landmark);
	}
	return associated_landmarks;
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   std::vector<LandmarkObs> observations, Map map_landmarks) {
//    cout << "test: " << MultiVariateGaussian(6, 3, 5, 3, .3, .3) << endl;
	for (int i = 0; i < num_particles; i++) {
		auto &particle = particles[i];

		std::vector<LandmarkObs> predicted; // THANKS TO THOMAS ANTHONY FOR HIS PINNED OVERVIEW FOR THIS.
		for (const auto &landmark : map_landmarks.landmark_list) {
			if (dist(landmark.x_f, landmark.y_f, particle.x, particle.y) <= sensor_range) {
				LandmarkObs obs{landmark.id_i, landmark.x_f, landmark.y_f};
				predicted.push_back(obs);
			}
		}

		// 1: Rotate and translate observations into particle-map coordinates
		vector<LandmarkObs> transformed;
		for (const auto &obs: observations) {
			LandmarkObs trans_obs;
			auto theta = particle.theta;
			trans_obs.id = obs.id;
			trans_obs.x = particle.x + (cos(theta) * obs.x - sin(theta) * obs.y);
			trans_obs.y = particle.y + (sin(theta) * obs.x + cos(theta) * obs.y);
			transformed.push_back(trans_obs);
		}

		// 2: Find nearest landmark on map, within sensor range distance, for each obs
		auto associated_landmarks = FindDataAssociations(predicted, transformed);

		particles[i].weight = 1.;

		// 3: Update particle weight by multiplying extent probability with probabilty of present observation
		for (int k = 0; k < transformed.size(); k++) {
			double landmark_proba = BivariateGaussian(associated_landmarks[k].x, associated_landmarks[k].y,
													  transformed[k].x, transformed[k].y, std_landmark[0], std_landmark[1]);
			particles[i].weight *= landmark_proba;
		}
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {     // TODO: resample, maybe use ring method...
	vector<Particle> temp_particles;
	discrete_distribution<> resampler(weights.begin(), weights.end());

	for (int i = 0; i < num_particles; i++) {
		temp_particles.push_back(particles[resampler(gen)]);
	}
	particles = temp_particles;
}


Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y) {
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}


string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}


string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}


string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
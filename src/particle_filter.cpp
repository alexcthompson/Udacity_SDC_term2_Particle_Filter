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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = 100;

    // set up random generators w/ mean = position data, std deviation as specified
    generator.seed(time(0));
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {
        Particle particle;
        particle.id = i;
        particle.x = dist_x(generator);
        particle.y = dist_y(generator);
        particle.theta = dist_theta(generator);
        particle.weight = 1.0;

        particles.push_back(particle);
        weights.push_back(particle.weight);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // set up random generators w/ mean 0.0, std deviation as specified
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for (int i = 0; i < num_particles; i++) {
        // update particle location based on position, theta, yaw_rate, and velocity
        // check that yaw_rate is not effectively 0
        Particle& particle = particles[i];

        if (abs(yaw_rate) >= 0.0001) {
            double theta_f = particle.theta + yaw_rate * delta_t;
            particle.x = particle.x + (velocity / yaw_rate) * (sin(theta_f) - sin(particle.theta));
            particle.y = particle.y + (velocity / yaw_rate) * (cos(particle.theta) - cos(theta_f));
            particle.theta = theta_f;
        }
        else {
            particle.x = particle.x + velocity * delta_t * cos(particle.theta);
            particle.y = particle.y + velocity * delta_t * sin(particle.theta);
            // theta is unchanged
        }

        // add noise to position and theta
        particle.x += dist_x(generator);
        particle.y += dist_y(generator);
        particle.theta += dist_theta(generator);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations,
                                     std::vector<double>& distances,
                                     std::vector<int>& associations,
                                     std::vector<double>& sense_x,
                                     std::vector<double>& sense_y) {
    // loop through observed, and for each find the nearest predicted, and change the obs id to that
    // added a lot to this function: passing distances, associations, sense_x, and sense_y since
    // it didn't make sense to do all that work twice
    for (int i = 0; i < observations.size(); i++) {
        double min_dist = numeric_limits<double>::max();

        double obs_x = observations[i].x;
        double obs_y = observations[i].y;
        int new_obs_id = -1;

        // loop over predicted to find minimum distance, tracking the landmark id
        for (int j = 0; j < predicted.size(); j++) {

            double pred_x = predicted[j].x;
            double pred_y = predicted[j].y;
            int pred_id = predicted[j].id;

            double op_dist = dist(obs_x, obs_y, pred_x, pred_y);

            if (op_dist < min_dist) {
                min_dist = op_dist;
                new_obs_id = pred_id;
            }
        }
        // update the landmark id
        observations[i].id = new_obs_id;
        distances.push_back(min_dist);
        associations.push_back(new_obs_id);
        sense_x.push_back(obs_x);
        sense_y.push_back(obs_y);
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // for normalization
    double sum_weights = 0.0;

    // set up random generators w/ mean 0.0, std deviation as specified
    normal_distribution<double> range_err_x(0, std_landmark[0]);
    normal_distribution<double> range_err_y(0, std_landmark[1]);

    // loop through particles
    for (int i = 0; i < num_particles; i++) {
        // particle info
        Particle& particle = particles[i];
        double x = particle.x;
        double y = particle.y;
        double theta = particle.theta;

        // produce list of observations mapped to particle's point of view
        vector<LandmarkObs> obs_in_particle_frame;

        for (int i = 0; i < observations.size(); i++) {
            LandmarkObs obs = observations[i];
            LandmarkObs transformed;
            
            // to avoid confusion regarding association, will change at association step
            transformed.id = -1;

            // rotate then add particle portion
            transformed.x = x + cos(theta) * obs.x - sin(theta) * obs.y;
            transformed.y = y + sin(theta) * obs.x + cos(theta) * obs.y;

            obs_in_particle_frame.push_back(transformed);
        }

        // get particle's actual observations within sensor range
        vector<LandmarkObs> predicted_in_particle_frame;

        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            float landmark_x = map_landmarks.landmark_list[j].x_f;
            float landmark_y = map_landmarks.landmark_list[j].y_f;
            int landmark_id = map_landmarks.landmark_list[j].id_i;

            if (dist(x, y, landmark_x, landmark_y) < sensor_range) {
                LandmarkObs sensed_landmark = LandmarkObs{landmark_id, landmark_x, landmark_y};
                predicted_in_particle_frame.push_back(sensed_landmark);
            }
        }

        // associate the observations with particle's landmarks
        vector<double> distances;
        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;
        dataAssociation(predicted_in_particle_frame, obs_in_particle_frame,
                        distances, associations, sense_x, sense_y);

        // compute likelihood
        particle.weight = 1.0;

        for (int j = 0; j < obs_in_particle_frame.size(); j++) {
            // we can drop gaussian norm [it's the same for all weights] and look simply at
            // distance squared over 2 * sigma_x**2, since sigma_x == sigma_y
            double obs_dist = distances[j];
            particle.weight *= exp(- (obs_dist * obs_dist) / (std_landmark[0] * std_landmark[0]));
        }
        weights[i] = particle.weight;
        sum_weights += particle.weight;

        SetAssociations(particle, associations, sense_x, sense_y);
    }

    // normalize weights
    for (int i = 0; i < num_particles; i++) {
        particles[i].weight = particles[i].weight / sum_weights;
        weights[i] = particles[i].weight;
    }

}

void ParticleFilter::resample() {
    // build the wheel for resampling
    vector<double> cum_weights;

    double sum_so_far = 0.0;
    for (int i = 0; i < num_particles; i++) {
        sum_so_far += weights[i];
        cum_weights.push_back(sum_so_far);
    }

    // RESAMPLE STEPS
    // new particle list
    vector<Particle> new_particles;

    // get first spoke
    uniform_real_distribution<double> unif(0.0, 1.0 / num_particles);
    double spoke_location = unif(generator);
    int i = 0;

    // move through spokes
    while (spoke_location <= 1.0) {
        if (cum_weights[i] <= spoke_location) {
            i++;
        }
        else {
            new_particles.push_back(particles[i]);
            spoke_location += 1.0 / num_particles;
        }
    }

   particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int> associations, 
                                     const std::vector<double> sense_x, const std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
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

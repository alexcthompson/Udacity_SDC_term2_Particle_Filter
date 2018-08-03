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
    num_particles = 10;

    // set up random generators w/ mean = position data, std deviation as specified
    default_random_engine generator;
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
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    cout << "\n\n==== PREDICTION STEP ====\n\n";
    // cout << "Cur delta_t:  " << delta_t << endl;
    // cout << "Cur std_pos:  " << std_pos[0] << ", " << std_pos[1] << ", " << std_pos[2] << endl;
    cout << "Cur velocity: " << velocity << endl;
    cout << "Cur yaw_rate: " << yaw_rate << endl;

    // set up random generators w/ mean 0.0, std deviation as specified
    default_random_engine generator;
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for (int i = 0; i < num_particles; i++) {
        // update particle location based on position, theta, yaw_rate, and velocity
        // check that yaw_rate is not effectively 0
        Particle particle = particles[i];
        // cout << "\n\nCurrent particle stats for " << particle.id << ":\n";
        // cout << "(" << particle.x << ", " << particle.y << ", " << particle.theta << ")" << endl;

        if (abs(yaw_rate) >= 0.0001) {
            double theta_f = particle.theta + yaw_rate * delta_t;
            particle.x = particle.x + (velocity / yaw_rate) * (sin(theta_f) - sin(particle.theta));
            particle.y = particle.y + (velocity / yaw_rate) * (cos(particle.theta) - cos(theta_f));
            particle.theta = theta_f;
        }
        else {
            particle.x = particle.x + velocity * cos(particle.theta);
            particle.y = particle.y + velocity * sin(particle.theta);
            // theta is unchanged
        }

        // add noise to position and theta
        particle.x += dist_x(generator);
        particle.y += dist_y(generator);
        particle.theta += dist_theta(generator);

        // cout << "\n\nNew particle stats for" << particle.id << ":\n";
        // cout << "(" << particle.x << ", " << particle.y << ", " << particle.theta << ")" << endl;
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

    // loop through observed, and for each find the nearest predicted, and change the obs id to that
    for (int i = 0; i < observations.size(); i++) {
        double min_dist = numeric_limits<double>::max();

        double obs_x = observations[i].x;
        double obs_y = observations[i].y;
        int new_obs_id = -1;

        // loop over predicted to find minimum distance, tracking the landmark id
        for (int j = 0; j < predicted.size(); j++) {
            double pred_x = predicted[i].x;
            double pred_y = predicted[i].y;
            int pred_id = predicted[i].id;

            double op_dist = dist(obs_x, obs_y, pred_x, pred_y);

            if (op_dist < min_dist) {
                min_dist = op_dist;
                new_obs_id = pred_id;
            }

        // update the landmark id
        observations[i].id = new_obs_id;
        }
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
    cout << "observations.size()" << observations.size() << endl;

    // set up random generators w/ mean 0.0, std deviation as specified
    default_random_engine generator;
    normal_distribution<double> range_err_x(0, std_landmark[0]);
    normal_distribution<double> range_err_y(0, std_landmark[1]);

    // loop through particles
    for (int i = 0; i < num_particles; i++) {
        // particle info
        Particle particle = particles[i];
        double x = particle.x;
        double y = particle.y;
        double theta = particle.theta;

        cout << "== particle " << particle.id << " ==" << endl;
        cout << "(" << x << ", " << y << ", " << theta << ")" << endl << endl;

        // produce list of observations mapped to particle's point of view
        vector<LandmarkObs> obs_in_particle_frame;

        for (int i = 0; i < observations.size(); i++) {
            LandmarkObs obs = observations[i];
            // cout << "== observation " << i << " ==" << endl;
            // cout << "(" << obs.x << ", " << obs.y << ")" << endl << endl;

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

        for (int i = 0; i < map_landmarks.landmark_list.size(); i++) {
            float landmark_x = map_landmarks.landmark_list[i].x_f;
            float landmark_y = map_landmarks.landmark_list[i].y_f;
            int landmark_id = map_landmarks.landmark_list[i].id_i;

            if (dist(x, y, landmark_x, landmark_y) < sensor_range) {
                cout << "landmark loc: (" << landmark_x << ", " << landmark_y << ")" << endl;
                LandmarkObs sensed_landmark = LandmarkObs{landmark_id, landmark_x, landmark_y};
                predicted_in_particle_frame.push_back(sensed_landmark);
            }
        }

        // associate the observations with particle's landmarks
        dataAssociation(predicted_in_particle_frame, obs_in_particle_frame);

        // compute likelihood


    }

    // normalize weights

}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

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

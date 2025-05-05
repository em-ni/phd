#include "Skeleton.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <mutex>

// TODOE: ATE is too similar among branches
namespace ORB_SLAM3 {

Skeleton::Skeleton(string &referenceCenterlinePath, Atlas* pAtlas)
    : mReferenceCenterlinePath(referenceCenterlinePath), mpAtlas(pAtlas) {
    // Load the reference centerline poses
    SetReferenceCenterline();

    // Init TCP server
    serverSocket = InitServer();
    if (serverSocket < 0) {
        std::cerr << "Error initializing server." << std::endl;
    }
    else {
        std::cout << "Server initialized successfully." << std::endl;
        connected = true;
    }
}

Skeleton::~Skeleton() {
    // Cleanup resources if needed.
}

void Skeleton::Run() {
    while (true) {
        if (mStopRequested) {
            std::cout << "Stop requested in Skeleton." << std::endl;
            break;
        }
        bool isDebug = false;
        if (isDebug) std::cout << "Skeleton thread running..." << std::endl;
        if (connected) {
            // auto startTime = std::chrono::steady_clock::now();
            // Find candidate trajectories
            std::vector<std::vector<Sophus::SE3f>> candidateTrajectories = FindCandidateTrajectories();
            std::vector<Sophus::SE3f> bestCandidateTrajectory;
            int bestCandidateIndex = -1;
            double bestATE = std::numeric_limits<double>::max();
            Sophus::Sim3f bestSim3;
            Sophus::SE3f lastKFPose;

            // Get the current trajectory (coming from slam) 
            // TODOE: this must be the trajectory from the current pose to the origin not the complete trajectory
            Map* pActiveMap = mpAtlas->GetCurrentMap();
            if (!pActiveMap) {
                std::cerr << "No active map found." << std::endl;
                continue;
            }
            const vector<KeyFrame*> vpKFs = pActiveMap->GetAllKeyFrames();
            if (vpKFs.empty()) {
                if (isDebug) std::cerr << "No keyframes found." << std::endl;
                continue;
            }
            std::vector<Sophus::SE3f> currentTrajectory;
            for (size_t i = 0; i < vpKFs.size(); i++) {
                KeyFrame* pKF = vpKFs[i];
                Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
                Sophus::SE3f pose(Twc);
                currentTrajectory.push_back(pose);
                if (i == vpKFs.size() - 1) {
                    lastKFPose = pose;
                }
            }
            if (isDebug) std::cout << "Current trajectory poses: " << currentTrajectory.size() << std::endl;
            if (currentTrajectory.size() < 10) {
                if (isDebug) std::cout << "Current trajectory is too short." << std::endl;
                continue;
            }
            
            // Align current trajectory to each candidate trajectory
            for (size_t i = 0; i < candidateTrajectories.size(); i++) {
                std::vector<Sophus::SE3f> &candidateTrajectory = candidateTrajectories[i];
                if (candidateTrajectory.empty()) {
                    std::cerr << "Empty candidate trajectory found." << std::endl;
                    continue;
                }

                // Align the current trajectory to the candidate trajectory
                Sophus::Sim3f sim3 = AlignTrajectories(candidateTrajectory, currentTrajectory, false, false);

                // Compute the ATE
                double ate = CalculateATE(candidateTrajectory, currentTrajectory, sim3);
                if (ate < bestATE) {
                    bestATE = ate;
                    bestCandidateIndex = i;
                    bestSim3 = sim3;
                }
                if (isDebug) std::cout << "Candidate trajectory index: " << i << " ATE: " << ate << std::endl;
            }

            // Select the one with the smallest ATE
            if (bestCandidateIndex >= 0) {
                bestCandidateTrajectory = candidateTrajectories[bestCandidateIndex];
                if (isDebug) std::cout << "\nBest candidate trajectory found at index: " << bestCandidateIndex << " with ATE: " << bestATE << std::endl;
            }
            else {
                if (isDebug) std::cout << "No best candidate trajectory found." << std::endl;
            }

            // Transform the last keyframe pose according to the best similarity transformation
            Eigen::Matrix3f R_new = bestSim3.rotationMatrix() * lastKFPose.rotationMatrix();
            Eigen::Vector3f t_new = bestSim3.scale() * (bestSim3.rotationMatrix() * lastKFPose.translation()) + bestSim3.translation();
            SetCurPose(Sophus::SE3f(R_new, t_new));
            // std::cout << "Last keyframe pose: " << lastKFPose.translation().transpose() << std::endl;
            // std::cout << "Aligned last keyframe pose: " << mCurPose.translation().transpose() << std::endl;

            // Send the pose to the server
            SendPose();
        }
        else {
            if (isDebug) std::cerr << "Server not connected. Skeleton not running" << std::endl;
        }
        
        // Sleep for a while
        // auto endTime = std::chrono::steady_clock::now();
        // auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        // if (isDebug) std::cout << "Skeleton thread finished in: " << elapsedTime.count() << " ms" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    SetStopped();

}

void Skeleton::RequestStop() {
    std::unique_lock<std::mutex> lock(mMutexStopRequested);
    mStopRequested = true;
}

bool Skeleton::IsStopped() {
    std::unique_lock<std::mutex> lock(mMutexStopped);
    return isStopped;
}

void Skeleton::SetStopped() {
    std::unique_lock<std::mutex> lock(mMutexStopped);
    isStopped = true;
}

template<typename MatrixType>
std::string MatrixToString(const MatrixType& m) {
    std::stringstream ss;
    ss << m;
    return ss.str();
}

void Skeleton::SendPose() {
    bool isDebug = false;
    // Get the current pose
    auto curPose = GetCurPose().matrix();
    std::string data = MatrixToString(curPose);
    if (send(serverSocket, data.c_str(), data.size(), 0) < 0) {
        std::cerr << "Error sending data." << std::endl;
    }
    else {
        if (isDebug) std::cout << "Data sent successfully." << std::endl;
    }    
}

int Skeleton::InitServer(){
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        std::cerr << "Error opening socket." << std::endl;
        return -1;
    }

    struct sockaddr_in serv_addr;
    memset(&serv_addr, '0', sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(serverPort);

    if (inet_pton(AF_INET, serverIP.c_str(), &serv_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address/Address not supported." << std::endl;
        return -1;
    }

    if (connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "Connection failed." << std::endl;
        return -1;
    }

    return sockfd;
}

Sophus::SE3f Skeleton::GetCurPose() {
    std::unique_lock<std::mutex> lock(mMutexCurPose);
    return mCurPose;
}

void Skeleton::SetCurPose(Sophus::SE3f pose) {
    std::unique_lock<std::mutex> lock(mMutexCurPose);
    mCurPose = pose;
}

void Skeleton::SetCurvilinearAbscissa(double value) {
    std::unique_lock<std::mutex> lock(mMutexCurvilinearAbscissa);
    mCurvilinearAbscissa = value;
}

double Skeleton::GetCurvilinearAbscissa() {
    std::unique_lock<std::mutex> lock(mMutexCurvilinearAbscissa);
    return mCurvilinearAbscissa;
}

void Skeleton::SetReferenceCenterline() {

    // Assume the centerlines are saved as a series of pose Twc in TUM format: timestamp px py pz qx qy qz qw
    // in a single .txt file for each branch
    
    // Protect with mutex
    std::unique_lock<std::mutex> lock(mMutexRefCenterlinePoses);

    // Clear any existing reference centerline poses.
    mRefCenterlinePoses.clear();

    bool isDebug = false;

    int branchId = 1;
    while (true){
        // Build the full name: e.g. <input_folder>/b1_tum.txt , <input_folder>/b2_tum.txt etc.
        std::string branchFileName = mReferenceCenterlinePath + "/b" + std::to_string(branchId) + "_tum.txt";
        if (isDebug) std::cout << "Reading branch file: " << branchFileName << std::endl;

        // Try to open the file.
        std::ifstream branchFile(branchFileName);
        if (!branchFile.is_open()) {
            if (isDebug) std::cout << "No more centerline found after index: " << branchId << std::endl; 
            break;
        }

        // Temporary container for the current branch poses
        std::vector<Sophus::SE3f> branchPoses;

        // Read the file line by line.
        std::string line;
        while (std::getline(branchFile, line)) {
            std::istringstream iss(line);
            double timestamp, px, py, pz, qx, qy, qz, qw;
            if (!(iss >> timestamp >> px >> py >> pz >> qx >> qy >> qz >> qw)) {
                if (isDebug) std::cout << "Error reading line: " << line << std::endl;
                break;
            }

            // Create a SE3 pose from the read values.
            Eigen::Quaternionf q(qw, qx, qy, qz);
            Eigen::Vector3f t(px, py, pz);
            Sophus::SE3f pose(q, t);

            // Add the pose to the branch poses.
            branchPoses.push_back(pose);
        }

        // Close this branch file.
        branchFile.close();

        // Add the branch poses to the reference centerline poses.
        mRefCenterlinePoses.push_back(branchPoses);

        if (isDebug) std::cout << "Branch " << branchId << " poses: " << branchPoses.size() << std::endl;

        // Move to the next branch.
        branchId++;
    }

    // Print the number of branches and poses.
    std::cout << "Total branches centerline loaded: " << mRefCenterlinePoses.size() << std::endl;

}

std::vector<std::vector<Sophus::SE3f>> Skeleton::GetReferenceCenterline() {
    std::unique_lock<std::mutex> lock(mMutexRefCenterlinePoses);
    return mRefCenterlinePoses;
}

std::vector<std::vector<Sophus::SE3f>> Skeleton::FindCandidateTrajectories() {

    bool isDebug = false;

    std::vector<std::vector<Sophus::SE3f>> candidateTrajectories;

    // Get the reference centerline poses
    std::vector<std::vector<Sophus::SE3f>> refCenterlinePoses = GetReferenceCenterline();
    if (refCenterlinePoses.empty()) {
        std::cerr << "No reference centerline poses found." << std::endl;
        return candidateTrajectories;
    }

    // Check if all branches are empty
    bool allBranchesEmpty = true;
    for (auto &branchPoses : refCenterlinePoses) {
        if (!branchPoses.empty()) {
            allBranchesEmpty = false;
            break;
        }
    }
    if (allBranchesEmpty) {
        std::cerr << "All branches are empty." << std::endl;
        return candidateTrajectories;
    }

    // Get the current curvilinear abscissa
    double curvilinearAbscissa = GetCurvilinearAbscissa();

    // Loop over each branch and find the candidate pose for each 
    for (size_t b = 0; b < refCenterlinePoses.size(); b++) {
        std::vector<Sophus::SE3f> &branchPoses = refCenterlinePoses[b];
        if (branchPoses.empty()) {
            if (isDebug) std::cout << "Branch " << b << " is empty." << std::endl;
            continue;
        }

        // Find the candidate pose for this branch by projecting the c.a. onto the centerline of the branch
        Sophus::SE3f candidatePose;
        size_t candidatePoseIndex = 0;
        double minDist = std::numeric_limits<double>::max();
        double s_i = 0.0;
        bool candidateFound = false;

        for (size_t i = 1; i < branchPoses.size(); i++) {
            Eigen::Vector3d p1 = branchPoses[i-1].translation().cast<double>();
            Eigen::Vector3d p2 = branchPoses[i].translation().cast<double>();

            // Compute the distance between two consecutive points
            double dist = (p2 - p1).norm();

            // Add the distance to the current s_i
            s_i += dist;

            // Compute the difference between the current c.a. and the real one
            double diff = std::abs(curvilinearAbscissa - s_i);
            if (diff < minDist) {
                minDist = diff;
                candidatePose = branchPoses[i];
                candidatePoseIndex = i;
                candidateFound = true;
            }
            // TODOE: break if the distance is increasing
        }
        if (!candidateFound) {
            std::cerr << "No candidate pose found for branch: " << b << std::endl;
            continue;
        }
        else{
            // Build the candidate trajectory for this branch taking all the poses from the beginning to the candidate pose
            std::vector<Sophus::SE3f> candidateTrajectory;
            for (size_t i = 0; i <= candidatePoseIndex; i++) {
                candidateTrajectory.push_back(branchPoses[i]);
            }
            candidateTrajectories.push_back(candidateTrajectory);
        }
    }
    if (isDebug) std::cout << "Candidate trajectories found: " << candidateTrajectories.size() << std::endl;
    return candidateTrajectories;
}

std::vector<Eigen::Vector3d> Skeleton::ResampleTrajectory(const std::vector<Sophus::SE3f>& trajectory, size_t num_samples)
{
    std::vector<Eigen::Vector3d> resampled;
    if (trajectory.empty())
        return resampled;
    size_t N = trajectory.size();
    std::vector<double> cum_dist(N, 0.0);
    cum_dist[0] = 0.0;
    // Compute cumulative arc-length
    for (size_t i = 1; i < N; i++) {
        Eigen::Vector3d p1 = trajectory[i-1].translation().cast<double>();
        Eigen::Vector3d p2 = trajectory[i].translation().cast<double>();
        double d = (p2 - p1).norm();
        cum_dist[i] = cum_dist[i-1] + d;
    }
    double total_length = cum_dist.back();
    // If total length is zero, all poses are identical.
    if (total_length <= 0.0) {
        resampled.resize(num_samples, trajectory[0].translation().cast<double>());
        return resampled;
    }
    resampled.resize(num_samples);
    // Uniformly sample along the arc-length.
    for (size_t i = 0; i < num_samples; i++) {
        double target = total_length * i / (num_samples - 1);
        // Find the segment in which 'target' lies.
        size_t idx = 0;
        while (idx < N - 1 && cum_dist[idx + 1] < target)
            idx++;
        if (idx == N - 1) {
            resampled[i] = trajectory.back().translation().cast<double>();
        } else {
            double seg_len = cum_dist[idx + 1] - cum_dist[idx];
            double ratio = (target - cum_dist[idx]) / seg_len;
            Eigen::Vector3d p1 = trajectory[idx].translation().cast<double>();
            Eigen::Vector3d p2 = trajectory[idx + 1].translation().cast<double>();
            resampled[i] = p1 + ratio * (p2 - p1);
        }
    }
    return resampled;
}

Sophus::Sim3f Skeleton::AlignTrajectories(const std::vector<Sophus::SE3f>& model,
                                            const std::vector<Sophus::SE3f>& data,
                                            bool known_scale, bool yaw_only)
{
    // Choose a fixed number of points for resampling.
    const size_t num_samples = std::min(model.size(), data.size());
    std::vector<Eigen::Vector3d> modelPoints = ResampleTrajectory(model, num_samples);
    std::vector<Eigen::Vector3d> dataPoints  = ResampleTrajectory(data, num_samples);
    size_t n = num_samples;
    
    // Compute centroids.
    Eigen::Vector3d mu_model = Eigen::Vector3d::Zero();
    Eigen::Vector3d mu_data  = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < n; i++) {
        mu_model += modelPoints[i];
        mu_data  += dataPoints[i];
    }
    mu_model /= static_cast<double>(n);
    mu_data  /= static_cast<double>(n);
    
    // Build matrices with each row corresponding to a centered 3D point.
    Eigen::MatrixXd M(n, 3);  // for model trajectory
    Eigen::MatrixXd D(n, 3);  // for data trajectory
    for (size_t i = 0; i < n; i++) {
        M.row(i) = modelPoints[i].transpose() - mu_model.transpose();
        D.row(i) = dataPoints[i].transpose()  - mu_data.transpose();
    }
    
    // Compute covariance matrix.
    Eigen::Matrix3d C = (M.transpose() * D) / static_cast<double>(n);
    
    // Variance of data points.
    double sigma2 = D.array().square().sum() / static_cast<double>(n);
    if (sigma2 < 1e-12) {
        std::cerr << "Sigma2 is too small, returning identity transform." << std::endl;
        return Sophus::Sim3f();
    }
    
    // SVD of covariance matrix.
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(C, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    // Create a reflection-correcting diagonal matrix.
    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    if (U.determinant() * V.determinant() < 0)
        S(2, 2) = -1;
    
    Eigen::Matrix3d R;  // rotation matrix
    if (!yaw_only) {
        // Full 3D rotation.
        R = U * S * V.transpose();
    } else {
        // Restrict rotation to yaw (rotation about the Z-axis).
        Eigen::Matrix3d rot_C = D.transpose() * M;
        double theta = GetBestYaw(rot_C);
        R = RotZ(theta);
    }
    
    // Compute scale.
    double s;
    if (known_scale)
        s = 1.0;
    else {
        Eigen::Vector3d singularValues = svd.singularValues();
        s = (singularValues(0) * S(0,0) +
             singularValues(1) * S(1,1) +
             singularValues(2) * S(2,2)) / sigma2;
    }
    
    // Use a threshold to avoid a degenerate (tiny) scale.
    float s_f = static_cast<float>(s);
    if (s_f < 1e-3f) {
        std::cerr << "Warning: computed scale factor is too small (" << s_f
                  << "). Setting scale to 1.0." << std::endl;
        s_f = 1.0f;
    }
    
    // Compute translation.
    Eigen::Vector3d t = mu_model - s * R * mu_data;
    
    // Return the similarity transformation using the RxSO3 pattern.
    return Sophus::Sim3f(
        Sophus::RxSO3d(s_f, R).cast<float>(),
        t.cast<float>());
}

double Skeleton::CalculateATE(const std::vector<Sophus::SE3f>& model,
                                const std::vector<Sophus::SE3f>& data,
                                const Sophus::Sim3f& sim3)
{
    const size_t num_samples = 50;
    std::vector<Eigen::Vector3d> modelPoints = ResampleTrajectory(model, num_samples);
    std::vector<Eigen::Vector3d> dataPoints  = ResampleTrajectory(data, num_samples);
    
    if (modelPoints.empty() || dataPoints.empty() || modelPoints.size() != dataPoints.size()){
        std::cerr << "Trajectory size mismatch or empty trajectories!" << std::endl;
        return -1;
    }
    
    double sum_squared_error = 0.0;
    for (size_t i = 0; i < modelPoints.size(); i++) {
        // Convert to float for transformation.
        Eigen::Vector3f p_model = modelPoints[i].cast<float>();
        Eigen::Vector3f p_data  = dataPoints[i].cast<float>();
        // Transform the data point using the similarity transform.
        Eigen::Vector3f p_aligned = sim3 * p_data;
        double err = (p_model - p_aligned).squaredNorm();
        sum_squared_error += err;
    }
    double rmse = std::sqrt(sum_squared_error / modelPoints.size());
    return rmse;
}

double Skeleton::GetBestYaw(const Eigen::Matrix3d& C)
{
    double A = C(0, 1) - C(1, 0);
    double B = C(0, 0) + C(1, 1);
    return M_PI / 2 - std::atan2(B, A);
}


Eigen::Matrix3d Skeleton::RotZ(double theta)
{
    Eigen::Matrix3d R;
    R << std::cos(theta), -std::sin(theta), 0,
         std::sin(theta),  std::cos(theta), 0,
         0,                0,               1;
    return R;
}


} // namespace ORB_SLAM3
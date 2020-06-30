#!/usr/bin/zsh

echo "STARTING SIMULATION"
nohup roslaunch torobo_bringup tabletop.launch sim:=true gazebo_gui:=false > out/torobo.out &
echo "$!" > out/server.pid
sleep 10
echo "RELAYING OUTPUT"
nohup rosrun topic_tools relay /gazebo/model_states /model_states > out/relay.out &
echo "$!" >> out/server.pid
echo "SETTING PHYSICS"
rosservice call /gazebo/set_physics_properties "{time_step: 0.002, max_update_rate: -1, gravity: {x: 0.0, y: 0.0, z: -9.8}, ode_config: {auto_disable_bodies: False, sor_pgs_precon_iters: 0, sor_pgs_iters: 50, sor_pgs_w: 1.3, sor_pgs_rms_error_tol: 0.0, contact_surface_layer: 0.001, contact_max_correcting_vel: 100.0, cfm: 0.0, erp: 0.2, max_contacts: 20}}"

package com.griddynamics.capstone_project.controller;

import com.griddynamics.capstone_project.service.AudioAuthService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/audio")
public class AudioAuthController {

    @Autowired
    private AudioAuthService audioAuthService;

    @PostMapping("/authenticate")
    public ResponseEntity<String> predictUser(@RequestParam("file") MultipartFile file) {
        try {
            String response = audioAuthService.predictUser(file);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.status(500).body(e.getMessage());
        }
    }
}

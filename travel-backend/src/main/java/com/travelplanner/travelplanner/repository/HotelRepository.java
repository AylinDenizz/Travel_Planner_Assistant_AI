package com.travelplanner.travelplanner.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;

import javax.swing.text.html.parser.Entity;
import java.util.Optional;
import java.util.UUID;


public interface HotelRepository extends JpaRepository<Entity, Long>, JpaSpecificationExecutor<Entity> {
    Optional<Entity> findByUuid(UUID uuid) ;
}

